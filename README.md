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
enforce and record           the system acts, the model only advises
```

The policy engine is the one step with no model in it. Refunds carry legal and
financial consequences, so they get plain rules that behave the same way every
time. Whether a reply is safe to send is a judgement call, and that part is the
LLM's job.

If the order number is missing or wrong, the lookup comes back empty and the
policy engine asks the customer for it. A model mistake costs someone a round
trip. It can't hand them somebody else's order.

There are two things the drafter deliberately can't see. One is the routing
guidelines - a drafter that knows the rules writes replies that pass them, and
then the router isn't measuring anything. The other is the customer's history. A
fraud flag should change whether a human checks the reply, not how politely it's
worded.

## Three modes

The same router, changing only how much of the guideline corpus it gets.

| Mode | Guidelines given |
|---|---|
| `prompt_only` | none, to see what the model manages unaided |
| `full_context` | all of them |
| `rag` | only the most relevant, retrieved by meaning |

## Running it

Needs [Ollama](https://ollama.com) with two local models:

```bash
ollama pull qwen3.5:4b        # the router
ollama pull nomic-embed-text  # guideline search
```

Then:

```bash
uv sync
uv run llmgov run --mode rag --top-k 10
```

`--mode` picks one of the three above. `--top-k` sets how many guidelines the
search hands over, and only applies to `rag`. 

Expect about half an hour for all 30 cases. Each one is three model calls -
reading the order number, writing the reply, judging it - and the model runs on
the CPU.

Results land in `runs/<n>-<mode>/`: `traces.jsonl` with a full record of every
case, and `metrics.json` with the scores. Traces are never rewritten. Each one
carries the prompt versions, the retrieval settings, and the Ollama version and
model digest that produced it, so an old run can be read back and understood
after everything else has moved on.


## The interface

```bash
uv run uvicorn llmgov.api.app:app --reload
```

Then open <http://127.0.0.1:8000>. It runs the same code as the CLI, in a browser.

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
  cases/        test cases with a gold label, what a careful reviewer would decide
  store/        SQLite - the order book and the customer accounts behind it
  policy/       the return policy, as configuration
  guidelines/   when a reply needs a human, as a policy PDF
  drafts/       replies pinned from one run, so later runs can reuse them
src/llmgov/
  policy/       the rule-based decision engine
  risk/         pattern detection and PII masking
  extraction/   reads the order number out of the customer's message
  drafting/     writes the reply from masked facts
  routing/      the router, its prompts, and guideline retrieval
  evaluation/   runs cases and scores them against the gold labels
  api/          the web interface and its templates
  loading_pdf.py  pulls the eighteen clauses out of the policy PDF
  store.py      reads the order book and the accounts out of SQLite
```

## Evaluation

Every decision is checked against a hand-written answer key.

The router can get a case wrong in two ways, and they don't cost the same.

Unsafe auto-send - a reply that needed a person went out on its own. 
Unnecessary review - a reply that was fine got held back. 
From those come coverage, how much went out without a human, and selective risk, how much of what went out shouldn't have. They only make sense as a pair. Send nothing and the risk is zero; send everything and the coverage is perfect.

AURC (area under the risk-coverage curve) measures how well the router knows when it might be wrong. Coverage and selective risk describe the one point where the router happened to draw its line between send and review. AURC looks at every point where it could have drawn it.

It's measured like this. For each case the router records how likely it thinks AUTO_SEND is, taken from the model's own output probabilities rather than a number it writes down. The cases are sorted by that, most confident first. Then, going down the list one case at a time, selective risk is worked out as if everything above that point had been sent. AURC is the average of all those values.

A low AURC means the cases the router was surest about were also the ones it got right, and its mistakes sat at the uncertain end. That's what makes a router usable in practice: you can pick a cut-off, send everything above it, and know roughly how much risk you're taking on.

Precision, recall, F1 and accuracy are reported alongside. 


## Future Work

A larger case set, then separate test and Validation sets for fine Tuning. Fine-tuning the model and
comparing it against Retrieval. 
