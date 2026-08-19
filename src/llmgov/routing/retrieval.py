from __future__ import annotations
import re

import faiss
import numpy as np
import ollama

from ..schemas import Case, Guideline, RetrievedGuideline, SystemFacts

DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_K = 5

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_MIN_PROBE = 12
RRF_K = 60


def case_query(case: Case) -> str:
    return _flat(case.user_message)


def case_probes(case: Case) -> list[str]:
    probes = [_flat(case.user_message)]
    probes += _sentences(case.user_message)
    probes += _sentences(case.draft_response)

    facts = _facts_probe(case.system_facts)
    if facts:
        probes.append(facts)

    seen: set[str] = set()
    out = []
    for p in probes:
        if len(p) >= _MIN_PROBE and p not in seen:
            seen.add(p)
            out.append(p)
    return out


class GuidelineIndex:
    def __init__(
        self,
        guidelines: list[Guideline],
        *,
        model_name: str = DEFAULT_EMBED_MODEL,
        host: str = DEFAULT_HOST,
    ) -> None:
        if not guidelines:
            raise ValueError("no guidelines to index")
        self.guidelines = guidelines
        self.model_name = model_name
        self.client = ollama.Client(host=host)

        vectors = self._embed([DOC_PREFIX + _flat(g.text) for g in guidelines])
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: str, k: int = DEFAULT_K) -> list[RetrievedGuideline]:
        return self.search_many([query], k=k)

    def search_many(
        self, queries: list[str], k: int = DEFAULT_K
    ) -> list[RetrievedGuideline]:
        if not queries:
            return []
        k = min(k, len(self.guidelines))
        vectors = self._embed([QUERY_PREFIX + q for q in queries])
        scores, positions = self.index.search(vectors, k)

        fused: dict[int, float] = {}
        for row_positions in positions:
            for rank, pos in enumerate(row_positions, start=1):
                if pos < 0:
                    continue
                fused[int(pos)] = fused.get(int(pos), 0.0) + 1.0 / (RRF_K + rank)

        top = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]
        return [
            RetrievedGuideline(
                guideline_id=self.guidelines[pos].guideline_id,
                text=_flat(self.guidelines[pos].text),
                score=round(score, 6),
                rank=rank,
            )
            for rank, (pos, score) in enumerate(top, start=1)
        ]

    def _embed(self, texts: list[str]) -> np.ndarray:
        response = self.client.embed(model=self.model_name, input=texts)
        vectors = np.array(response["embeddings"], dtype="float32")
        faiss.normalize_L2(vectors)
        return vectors


def _sentences(text: str) -> list[str]:
    return [s for s in (_flat(p) for p in _SENTENCE.split(text or "")) if s]


def _facts_probe(f: SystemFacts) -> str:
    bits = []
    if f.account_flags:
        bits.append("the account is flagged for " + ", ".join(f.account_flags))
    if f.refunds_last_12m is not None and f.total_orders:
        bits.append(
            f"{f.refunds_last_12m} refunds claimed in the last 12 months "
            f"across {f.total_orders} orders"
        )
    if f.order_id is None:
        bits.append("no order could be identified from the message")
    else:
        if f.order_status:
            bits.append("order status is " + f.order_status.lower().replace("_", " "))
        if f.return_condition:
            bits.append("the item came back " + f.return_condition.value)
        if f.is_faulty:
            bits.append("the item was faulty on arrival")
        if f.delivered_days_ago is None:
            bits.append("no delivery date is recorded")
        if f.category is None:
            bits.append("the product category is unknown")
    return "; ".join(bits)


def _flat(text: str) -> str:
    return " ".join(text.split())
