# llmgov

A RAG-LLM Routing Application for Human-in-the-Loop Customer-Service Automation

## What it does

This is a prototype for an automated customer service system in online retail,
handling refund and return requests. A customer writes asking about the status/possibility of refund from their order. The system looks up their order, applies the company's return policy
to work out whether the refund is granted, and a reply is drafted. Before that
reply goes out, this system decides whether it can be sent automatically or
whether a person should read it first.

Sending everything automatically is risky. A reply might repeat the customer's
bank details back to them, contradict the decision that was actually made, or
refuse someone who has just said they are speaking to a lawyer. But having a
person check every reply removes most of the benefit of automating. The question
this project asks is which replies actually need a human.

So for each drafted reply the system returns one of two answers:

> AUTO_SEND, or NEEDS_REVIEW.

It does not write the reply, and it does not decide the refund. It judges whether
what has already been decided and drafted is safe to release.

## Where the rules come from

The return policy lives in a configuration file rather than in code: how long the
return window is for each product category, how faulty goods are handled, when a
refund is reduced instead of refused. It follows published European retail
practice and the statutory minimums in the EU Consumer Rights Directive.

Separately there is a set of routing guidelines saying when a reply should go to
a person. Each one records where it came from, whether that is GDPR, the EU
consumer dispute framework, or simply company policy where the rule is our own
rather than a legal requirement.

## How a case flows through

```
customer message + order id
   ↓
look up the order            status, category, days since delivery, value
   ↓
policy engine                approve / partial / deny / request info
   ↓
mask personal data           IBAN, card, email, phone become placeholders
   ↓
retrieve guidelines          the rules most relevant to this case
   ↓
LLM router                   AUTO_SEND or NEEDS_REVIEW, with a written reason
   ↓
enforce and record           the system acts, the model only advises
```

Two decisions happen in different places, deliberately. The refund outcome is
decided by plain rules with no LLM involved, because refunds carry legal and
financial consequences and need to be consistent. Whether the reply
is safe to send is decided by the LLM, using the retrieved guidelines.

## Three modes

The same router, differing only in how much of the guideline corpus it receives.
Comparing them is the point of the research.

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

Results are written to `runs/<n>-<mode>/`, with a full trace for every case and
the scores. Runs are reproducible: temperature is zero and the seed is fixed, so
the same input always gives the same decision.

## Layout

```
data/
  cases/        test cases with a gold label, what a careful reviewer would decide
  orders/       order facts, keyed by order id
  policy/       the return policy, as configuration
  guidelines/   when a reply needs a human, with the source of each rule
src/llmgov/
  policy/       the rule-based decision engine
  risk/         pattern detection and PII masking
  routing/      the router, its prompts, and guideline retrieval
  evaluation/   runs cases and scores them against the gold labels
```

## Evaluation

Every decision is compared against a hand-written answer key. Two numbers matter
most. False negatives are risky replies that were sent anyway, which is the
error that causes real damage. The review rate is how much human work the system
creates. Driving false negatives to zero is easy if you send everything to a
person, but then nothing has been automated. Finding the balance is the research
question.


## Future Work
Many enhancements are planned for the future, including a nice UI with FastAPI backend, custom refund requests through the UI and LLM-generated
drafts rather than hard-coded, customer history feeding the decision, a larger case set with guidelines and rules as PDF's, fine-tuning the model, and splitting the components into separate services, stay tuned!

