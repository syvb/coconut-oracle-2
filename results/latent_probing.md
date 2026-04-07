# Latent Probing: Do CoDi's Continuous Thoughts Encode Numbers and Math?

## Motivation

CoDi (Continuous Distillation) replaces explicit chain-of-thought with 6 latent
reasoning steps in continuous embedding space. The model never generates text
during reasoning -- it loops hidden states through a projection layer and back
into the transformer. A natural question: are those latent representations
secretly encoding the numbers and operations that would appear in a normal CoT?

## Method

We run two experiments on 30 GSM8K test questions.

### Experiment 1: Top-k token decoding (`probe_latents.py`)

At each latent step, decode the hidden state two ways:
- **Cosine similarity to the token embedding matrix** -- what the projected
  latent vector "looks like" in embedding space.
- **LM head logits** -- what the model would predict as the next token from
  the pre-projection hidden state.

Check whether the top-k decoded tokens are numbers from the gold CoT solution.

### Experiment 2: Selectivity analysis (`probe_selectivity.py`)

A more rigorous test. Instead of just checking top-k, we compare the **full
logit distribution** over all 1,196 number tokens in the vocabulary:
- Partition number tokens into "gold" (appear in the CoT solution) vs "non-gold".
- Compare mean logit, median rank, and total probability mass.
- Do the same for math operator tokens (+, -, *, /).
- Track the rank of each equation's result number at each latent step.

## Results

### The embedding space is opaque

Cosine similarity to token embeddings yields essentially random tokens --
multilingual fragments, code tokens, no interpretable content. Only 2.3% of
top-10 tokens are numbers, with 0% overlap with gold CoT values. The
projection layer maps hidden states into a region of the 2048-dim space that
doesn't correspond to any natural tokens. This makes sense: the projection was
trained to produce useful transformer inputs, not human-readable text.

### The LM head reveals selective number encoding

When we decode through the LM head instead, gold CoT numbers are consistently
**boosted over non-gold numbers** at every latent step:

| Step | Gold mean logit | Non-gold mean logit | Delta | Gold median rank | Non-gold median rank |
|------|-----------------|---------------------|-------|------------------|----------------------|
| 1    | +0.61           | -0.82               | +1.43 | 20,163           | 38,326               |
| 2    | +1.43           | -1.71               | +3.13 | 16,550           | 54,693               |
| 3    | -0.42           | -1.80               | +1.38 | 19,685           | 37,407               |
| 4    | +0.71           | -1.73               | +2.45 | 18,865           | 53,641               |
| 5    | -0.08           | -1.10               | +1.02 | 19,079           | 33,509               |
| 6    | -0.45           | -0.94               | +0.48 | 34,811           | 43,027               |
| **ALL** | **+0.30**     | **-1.35**           | **+1.65** | **21,526**    | **43,434**           |

Gold numbers rank roughly 2x higher than non-gold numbers (median rank ~21k vs
~43k out of 128k vocabulary). The selectivity is strongest at **step 2**
(Delta=+3.13) and **step 4** (Delta=+2.45), with a 2-beat rhythm.

### Operators are also selectively boosted

Gold CoT operators (+, -, *, /) consistently have higher logits than non-gold
operators:

| Step | Gold op mean logit | Non-gold op mean logit | Delta |
|------|--------------------|------------------------|-------|
| 1    | 4.20               | 3.47                   | +0.73 |
| 2    | 4.50               | 4.02                   | +0.48 |
| 3    | 5.57               | 4.72                   | +0.86 |
| 4    | 3.99               | 3.28                   | +0.71 |
| 5    | 4.67               | 3.34                   | +1.33 |
| 6    | 3.24               | 2.40                   | +0.83 |

The operator effect peaks at **step 5** (Delta=+1.33), later than the number
peaks, suggesting numbers are established first and operators are resolved
after.

### Step 2 is the critical computation step

Tracking the rank of each equation's result number across latent steps reveals
that step 2 is where most computation happens:

| Equation index | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 | Step 6 |
|----------------|--------|--------|--------|--------|--------|--------|
| eq0 (1st)      | 494    | **5**  | 913    | 69     | 1,637  | 12,300 |
| eq1 (2nd)      | 430    | **50** | 2,492  | 197    | 1,828  | 18,910 |
| eq2 (3rd)      | 412    | 436    | 1,353  | **370**| 1,321  | 10,411 |
| eq3 (4th)      | 455    | **3**  | 1,342  | 1,119  | 6,144  | 22,056 |
| eq4 (5th)      | 1,274  | **104**| 3,724  | 782    | 8,916  | 10,389 |
| eq5 (6th)      | **151**| 88     | 172    | 5,044  | 343    | 21,670 |

A rank of 5 out of 128k tokens is remarkable -- the model strongly "knows" the
answer to the first equation at step 2. Most equation results peak at step 2,
with a secondary peak at step 4. The model does not process equations
sequentially (one per latent step) but instead computes most results in a
concentrated burst.

Later steps (5-6) see ranks degrade sharply across all equations, consistent
with the model transitioning from computation to answer formulation (top tokens
shift to "therefore", "thus", "$").

## Interpretation

1. **The latent space is not a simple calculator.** The projected embeddings
   are opaque -- they don't map to number or operator tokens in embedding
   space. The model's "thinking" is distributed across the hidden state in a
   way that isn't reducible to token-level representations.

2. **But the hidden states do encode computation-relevant numbers.** The LM
   head can partially decode them, and gold numbers are consistently and
   selectively boosted over irrelevant numbers. This isn't just "numbers are
   present" -- the *right* numbers for the problem are preferentially encoded.

3. **Computation is temporally concentrated, not sequential.** Step 2 is the
   primary computation step where most equation results peak. There's a
   secondary peak at step 4. The model doesn't use its 6 steps as 6 serial
   computation slots.

4. **Numbers peak before operators.** Number selectivity peaks at steps 2/4,
   operator selectivity at step 5. This suggests a compute-then-formulate
   progression within the latent reasoning loop.

## Reproducing

```bash
# Top-k token decoding (quick qualitative look)
python probe_latents.py -n 10 --json results/probe_results.json

# Full selectivity analysis
python probe_selectivity.py -n 30 --json results/probe_selectivity.json
```

## Data files

- `probe_results.json` -- per-question top-10 tokens at each step (10 questions)
- `probe_selectivity.json` -- per-question logit/rank stats for gold vs non-gold numbers and operators (30 questions)
