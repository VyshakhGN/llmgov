from __future__ import annotations
import faiss
import numpy as np
import ollama

from ..schemas import Case, Guideline, RetrievedGuideline

DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_K = 5

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def case_query(case: Case) -> str:
    return " ".join(case.user_message.split())


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
        k = min(k, len(self.guidelines))
        vector = self._embed([QUERY_PREFIX + query])
        scores, positions = self.index.search(vector, k)

        results = []
        for rank, (pos, score) in enumerate(zip(positions[0], scores[0]), start=1):
            g = self.guidelines[pos]
            results.append(
                RetrievedGuideline(
                    guideline_id=g.guideline_id,
                    text=_flat(g.text),
                    score=round(float(score), 4),
                    rank=rank,
                )
            )
        return results

    def _embed(self, texts: list[str]) -> np.ndarray:
        response = self.client.embed(model=self.model_name, input=texts)
        vectors = np.array(response["embeddings"], dtype="float32")
        faiss.normalize_L2(vectors)
        return vectors


def _flat(text: str) -> str:
    return " ".join(text.split())
