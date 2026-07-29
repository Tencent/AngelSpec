# Speculative Decoding

## Overview

Autoregressive LLM inference is memory-bandwidth bound: generating each token requires loading
the full model weights, so latency is dominated by memory traffic rather than compute.
Speculative decoding exploits this by letting a small **draft model** propose several tokens
ahead, which the large **target model** then verifies in a single parallel pass. Because
verifying many tokens costs about the same as generating one, accepting even a few speculated
tokens per step reduces latency.

## How it works

Two models cooperate:

- **Target model** — the large model you want to serve.
- **Draft model** — a small model trained to predict the next few tokens.

Each decoding round has three stages:

1. **Prefill** — the target model processes the prompt.
2. **Drafting** — the draft model proposes *N* candidate tokens. Being small, this is cheap.
3. **Verification** — the target model checks the *N* candidates in parallel. Accepted tokens
   are appended to the output; on the first rejection, the target model's own token is used and
   drafting resumes from there.

The verification step uses rejection sampling, so the output distribution is identical to
running the target model alone — speculative decoding changes *speed*, not *results*.

## Feature-space draft models

AngelSpec trains **feature-space** draft models (Eagle3 and its successors). Rather than
predicting from tokens alone, these drafters consume the target model's intermediate **hidden
states**, which carry richer information than the output logits. This makes the draft model more
accurate — and it is why training needs the target model's hidden states for every token, which
the [disaggregated architecture](disaggregated_architecture.md) is built to supply efficiently.

The draft architectures themselves — how they consume hidden states and how they predict — are
described in [The Draft-Model Family](draft_model_family.md).
