# llmgov

A RAG-LLM Routing Application for Human-in-the-Loop Customer-Service Automation

## What it does

A prototype customer service system for online retail, handling refunds and
returns. Someone writes in about a refund. The system finds the order number in
their message, looks it up, applies the return policy, and writes a reply. Then
it decides whether that reply can go out on its own or needs a person to read it
first.

That last decision is what the project is actually about.

Sending everything automatically is risky - a reply might quote the customer's
bank details back at them, or refuse someone who has just mentioned their lawyer.
Checking every reply by hand defeats the point of automating in the first place.
So which ones actually need a human?

For each draft the system answers:

> AUTO_SEND, or NEEDS_REVIEW.

The part making that call doesn't write the reply and doesn't decide the refund.
It only judges whether what has already been written is safe to send.

## Where the rules come from

The rules are based on things like return window lengths per item category. The numbers follow published European retail practice, with the EU
Consumer Rights Directive as the floor.

Routing guidelines are a separate thing. They say when a reply should go to a
person, and they sit in a PDF file. The
system parses the eighteen clauses out of it at startup. Each one records where
it came from - GDPR, the EU consumer dispute framework, or "company policy" where
we made the rule up ourselves. Nine of the eighteen cite regulation and the other
nine say so.

## How a case flows through

```
customer message
   ↓
read the order number        an LLM pulls it out of the text
   ↓
look up the order            status, category, days since delivery, value,
                             and the customer's history with us
   ↓
policy engine                approve / partial / deny / request info
   ↓
mask personal data           IBAN, card, email, phone partly hidden: DE89****00
   ↓
write the reply              an LLM drafts it from the masked facts
   ↓
retrieve guidelines          the rules most relevant to this case
   ↓
LLM router                   AUTO_SEND or NEEDS_REVIEW, with a written reason
   ↓
record the decision          cases marked for review appear in the review queue
```

The policy engine uses plain rules that behave the same way every time. Whether
a reply is safe to send is a judgement call, and that part is the LLM's job.

The [component diagram](docs/diagrams/component-diagram.svg) shows
how the baseline system fits together. The [sequence diagram](docs/diagrams/sequence.svg)
shows what happens step by step when a case runs from the web interface.

If the system finds no matching order, the policy engine asks the customer
for the order number.

There are two things the drafter deliberately can't see. One is the routing
guidelines - a drafter that knows the rules writes replies that pass them, and
then the router isn't measuring anything. The other is the customer's history. A
fraud flag should change whether a human checks the reply, not how politely it's
worded.

## Three modes

In the baseline, the same router can get a different amount of guidance.

| Mode | Guidelines given |
|---|---|
| `prompt_only` | none, to see what the model manages unaided |
| `full_context` | all of them |
| `rag` | only the most relevant, retrieved by meaning |

### Searcher and reply evaluator

By default, RAG builds search queries from the message, the draft, and the
order facts. The CLI also offers a version with two extra steps. A searcher
writes queries for the same guideline index. A reply evaluator checks the draft
against the policy decision, refund amount, and order facts.

The router still makes the send-or-review decision. It sees the draft, the
retrieved guidelines, and the evaluator's finding. If a step fails, the case
goes to review and the trace records which step failed.

## Running it

Needs [Ollama](https://ollama.com) with two local models:

```bash
ollama pull qwen3.5:4b        # language model
ollama pull nomic-embed-text  # guideline search
```

Then:

```bash
uv sync
uv run llmgov run --mode rag --top-k 10
```

`--mode` picks one of the three above. `--top-k` sets how many guidelines the
search hands over, and only applies to `rag`.

Use `--drafts data/drafts/v1.yaml` to reuse the saved replies from the 30
development cases. This keeps the replies the same across runs.

```bash
uv run llmgov run --mode rag --top-k 10 --drafts data/drafts/v1.yaml
```

For a quick check, add `--case-id C-001 --case-id C-014`. Run
`uv run llmgov run --help` to see the available `--architecture` settings.

Each run gets its own folder under `runs/`, with `traces.jsonl` for the case
records and `metrics.json` for the scores. The traces record the settings and
model builds used. When the searcher and evaluator are enabled, the traces also
show their outputs, timing, and token use.


## The interface

```bash
uv run uvicorn llmgov.api.app:app --reload
```

Then open <http://127.0.0.1:8000>. You can start baseline runs and inspect saved
runs in the browser.

The sidebar has the data on one side and the runs on the other.

**View rules, guidelines, cases, orders, customers** show what the system runs
on. Useful for seeing why a case came out the way it did: the order page puts the
delivery age next to the window that applies, and the customer page works out a
refund rate, so an account that returns half of what it buys stands out.

**View runs** lists every past run with its scores. Click one for the per-case
table, then click a case to see it go through the pipeline in order: the order
number that was read out of the message next to the one it should have been, the
facts, which rule fired and why, what got masked, the reply the model wrote, the
guidelines it was shown with their ranks, and the decision with its reason. The
exact prompt is behind a toggle at the bottom, so any decision can be replayed.

**Create a new run** picks a mode and any subset of the cases. Progress appears
as each case finishes and there is a stop button, which is worth having when a
full run takes half an hour. A stopped run still saves what it finished.

**The review queue** is reached from a run page. It shows the cases that were
flagged, with the customer's message, the facts, the decision, and the reply.
A reviewer can approve it, edit the wording, or reject it because the decision
itself is wrong. Those three answers say different things: an edit points at the
drafter, a rejection points at the policy. The router's own reasoning sits below
the buttons, collapsed, so reading it first does not colour the decision.

Reviews are written to `reviews.jsonl`.

## Layout

```
data/
  cases/        development cases and a draft batch of training cases
  store/        SQLite - the order book and the customer accounts behind it
  policy/       the return policy, as configuration
  guidelines/   when a reply needs a human, as a policy PDF
  drafts/       replies pinned from one run, so later runs can reuse them
src/llmgov/
  policy/       the rule-based decision engine
  risk/         pattern detection and PII masking
  extraction/   reads the order number out of the customer's message
  drafting/     writes the reply from masked facts
  routing/      the router, guideline retrieval, the searcher, and the reply evaluator
  evaluation/   runs cases and scores them against the gold labels
  api/          the web interface and its templates
  loading_pdf.py  pulls the eighteen clauses out of the policy PDF
  store.py      reads the order book and the accounts out of SQLite
docs/
  diagrams/     component and sequence diagrams (PlantUML sources and SVGs)
```

## Evaluation

Every decision is checked against a hand-written answer key.

The router can get a case wrong in two ways, and they don't cost the same.

- **Unsafe auto-send** means the router marked a reply for automatic send when
  it needed a human.
- **Unnecessary review** means the router held back a reply that was safe.

Coverage is the share marked for automatic send. Selective risk is the share
of those replies that should have gone to review. Read the two together.

AURC (area under the risk-coverage curve) measures how well the router knows when it might be wrong. Coverage and selective risk describe the one point where the router happened to draw its line between send and review. AURC looks at every point where it could have drawn it.

It's measured like this. For each case the router records how likely it thinks AUTO_SEND is, taken from the model's own output probabilities rather than a number it writes down. The cases are sorted by that, most confident first. Then, going down the list one case at a time, selective risk is worked out as if everything above that point had been sent. AURC is the average of all those values.

A lower AURC means the router did a better job of putting safe replies ahead
of risky ones in that ranking.

Precision, recall, F1 and accuracy are reported alongside.

The guideline search is scored on its own: for each case, did the search return the guideline
that decides it (hit rate), and how near the top (MRR). Every run also records the time and tokens
of each step, so the cost of each version is visible next to its results.

## Future Work

Review and expand the training cases, then set aside separate validation and
test cases. Fine-tune the router and compare it with the version that retrieves
guidelines.
