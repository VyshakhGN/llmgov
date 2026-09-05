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

The return policy sits in a config file rather than in code: window lengths per
category, how faulty goods are handled, when a refund gets reduced instead of
refused. The numbers follow published European retail practice, with the EU
Consumer Rights Directive as the floor.

Routing guidelines are a separate thing. They say when a reply should go to a
person. Each one records where it came from - GDPR, the EU consumer dispute
framework, or "company policy" where we made the rule up ourselves. Roughly half
are the latter, and the file says which.

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
search hands over, and only applies to `rag`. `--no-extract` takes the order
number from the case file instead of reading it from the message, which is
useful for measuring what extraction costs.

Expect about half an hour for all 30 cases. Each one is three model calls -
reading the order number, writing the reply, judging it - and the model runs on
the CPU.

Results land in `runs/<n>-<mode>/`: `traces.jsonl` with a full record of every
case, and `metrics.json` with the scores. Traces are never rewritten, and they
carry the prompt versions, so an old run can still be read back and understood
after the prompts have moved on.

Temperature is zero and the seed is fixed, so the same input gives the same
decision every time.

## The interface

```bash
uv run uvicorn llmgov.api.app:app --reload
```

Then open <http://127.0.0.1:8000>. It runs the same code as the CLI, in a browser.

The sidebar has the data on one side and the runs on the other.

**View rules, guidelines, cases, orders, customers** show the five data files.
Useful for seeing why a case came out the way it did: the order page puts the
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
  orders/       order facts, keyed by order id
  customers/    account age, orders placed, refund history, fraud flags
  policy/       the return policy, as configuration
  guidelines/   when a reply needs a human, with the source of each rule
src/llmgov/
  policy/       the rule-based decision engine
  risk/         pattern detection and PII masking
  extraction/   reads the order number out of the customer's message
  drafting/     writes the reply from masked facts
  routing/      the router, its prompts, and guideline retrieval
  evaluation/   runs cases and scores them against the gold labels
  api/          the web interface and its templates
```

## Evaluation

Every decision is checked against a hand-written answer key.

Two numbers matter. False negatives are the risky replies that got sent anyway,
which is the error that does real damage. Review rate is how much human work the
system creates. You can drive false negatives to zero by sending everything to a
person, but then nothing has been automated.

Order extraction is scored on its own. Each case records which order it is really
about, so whatever the model pulled out of the message can be checked against it.
That field never feeds into the case itself.

## Future Work

Several enhancements are planned, including a larger case set, fine-tuning the
model and comparing it against retrieval, guidelines that can be swapped and
re-indexed without a restart, and eventually splitting the components into
separate services. Stay tuned!
