# 1. Record architecture decisions

## Status

Accepted

## Context

This project makes a number of decisions that are easy to silently reverse
later (an embedding provider, a fusion strategy, a deploy shape) unless the
reasoning is written down next to the code, not just in a chat transcript or
a PR description that gets buried.

## Decision

We will use Architecture Decision Records (ADRs), one per significant
decision, stored in `docs/adr/` as numbered Markdown files following this
template (Context / Decision / Consequences).

## Consequences

Future contributors — including a future session of whoever is doing this
work — can see *why* a decision was made, not just what it is, without
re-deriving it from the diff. See `docs/plan.md` for the list of ADRs this
project expects to accumulate (RRF over score normalization, fastembed over
hosted embeddings, baking the index at build time, judge-free retrieval
metrics, embedding-namespaced collections, plain-httpx providers with
LangChain confined to the eval judge).
