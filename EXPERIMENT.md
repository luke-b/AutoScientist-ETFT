# ETFT Historical Replay Pilot

## Objective

Test the minimum falsifiable ETFT claim:

> **Does conditioning on historically realized innovation trajectories reduce the search effort needed to reconstruct a withheld improvement, relative to matched baselines?**

This pilot deliberately does **not** attempt to prove autonomous SOTA discovery. It first tests whether trajectory structure and causal transition metadata provide measurable incremental value on known historical progress.

## Hypothesis

Given the same base model, evaluator, candidate budget, token budget, and compute budget, ETFT conditioning should require fewer candidate evaluations to recover a held-out historical improvement reaching a predefined performance threshold.

A null or negative result is informative: if matched baselines perform equally well, historical ordering and/or causal transition metadata have not demonstrated incremental value.

## First target domain: SAT solver lineage

Initial target: a documented SAT-solver family such as MiniSat → Glucose → later lineage-linked descendants.

Why SAT:

- executable public artifacts exist;
- correctness and runtime are objectively measurable;
- the field contains explicit algorithmic innovations and descendants;
- evaluation can be CPU-first;
- the benchmark naturally matches evaluator-driven algorithm discovery.

The first curation pass should stay small: approximately **15–30 verified transitions**, not the full SAT competition archive.

## Experimental arms

All arms use the same model and evaluation harness.

### B0 — Final-state / unordered evidence

The model receives equivalent historical artifacts without chronological trajectory structure.

### B1 — Ordered trajectory

The model receives the ordered sequence of predecessors, but without causal transition metadata.

### ETFT — Ordered trajectory + causal Δ metadata

The model receives the ordered trajectory plus metadata describing, where recoverable:

- prior state;
- bottleneck / limitation;
- intervention;
- rationale;
- failed alternatives;
- measured outcome / performance delta.

An optional later arm may use machine-generated evolutionary trajectories as an additional baseline.

## Historical Replay protocol

For a selected transition:

```
A_i  --Δ_i-->  A_(i+1)
```

hide `A_(i+1)`, its implementation, and post-transition explanatory material.

Each arm starts from the same `A_i` and is allowed a fixed exploration budget. Generated candidates are executed by the same deterministic benchmark harness.

The test asks whether the hidden historical improvement can be reconstructed more efficiently.

## Pre-registered success criterion

For each held-out transition, define a target before running the experiment, for example:

- recover at least **90% of the historical performance gain**, or
- reach the held-out successor's benchmark score within a fixed tolerance.

Primary metric:

```
evaluations-to-threshold
```

Secondary metrics:

- tokens-to-threshold;
- wall-clock / compute-to-threshold;
- valid-candidate rate;
- replay success rate;
- functional correctness rate.

Proposed **Discovery Yield**:

```
Discovery Yield =
    threshold-crossing valid candidates
    -----------------------------------
          fixed exploration budget
```

Report distributions and medians across held-out transitions, not a single showcase result.

## Calibration → frontier test

A second-stage research question tests the ETFT stage-gate itself:

> Does Historical Replay calibration predict later frontier-search success?

For each run, measure a replay/calibration score `C_replay`, then run the same bounded frontier search and measure `Y_frontier` or success probability.

Test:

```
C_replay  <-->  frontier discovery success / yield
```

A useful correlation would support replay as an epistemic readiness signal. No correlation would argue against using reconstruction confidence as a gate.

## Leakage and contamination controls

Historical algorithms may appear in model pretraining data. The pilot should therefore:

1. withhold names and identifying comments where practical;
2. use transformed or obfuscated code variants;
3. avoid giving post-transition explanations to the replaying model;
4. include contamination probes;
5. prefer transitions that require functional reconstruction, not textual recollection;
6. treat semantic equivalence and benchmark performance as more important than source-code overlap.

## Minimal resource profile

The first pass should be CPU-first and should avoid expensive full-model fine-tuning.

Suggested order:

1. establish signal with matched prompting / retrieval conditions;
2. freeze the benchmark and evaluation protocol;
3. replicate across multiple held-out transitions;
4. only then test LoRA / fine-tuning if the representation itself shows signal.

## Deliverables

- curated trajectory records with provenance;
- deterministic benchmark harness;
- frozen baseline prompts / contexts;
- ETFT condition;
- structured run logs;
- evaluations-to-threshold curve;
- Discovery Yield comparison;
- short report including negative results.

## Decision rule

The pilot succeeds scientifically even if ETFT loses.

The important outcome is a reproducible answer to:

> **Does representing prior scientific/algorithmic progress as an ordered, causally annotated trajectory improve the efficiency of reconstructing known progress?**

Only after that question is answered should the project make a stronger claim about extrapolating beyond the known frontier.
