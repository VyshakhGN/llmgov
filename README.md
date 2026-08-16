# llmgov

A RAG-LLM Routing Application for Human-in-the-Loop Customer-Service Automation

## What it does

A prototype customer service system for online retail, handling refunds and
returns. Someone writes in about a refund. The system finds the order number in
their message, looks it up, applies the return policy, and writes a reply. Then
it decides whether that reply can go out on its own or needs a person to read it
first.

That last decision is what the project is actually about.

Sending everything automatically is risky — a reply might quote the customer's
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
person. Each one records where it came from — GDPR, the EU consumer dispute
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
guidelines — a drafter that knows the rules writes replies that pass them, and
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
uv run llmgov run --mode rag
```

Results land in `runs/<n>-<mode>/` — a full trace for every case, plus the
scores. Temperature is zero and the seed is fixed, so the same input gives the
same decision every time.

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

Several enhancements are planned, including a UI with a FastAPI backend for
browsing cases and working through the review queue, a larger case set,
fine-tuning the model and comparing it against retrieval, and eventually
splitting the components into separate services. Stay tuned!
