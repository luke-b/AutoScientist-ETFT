---
name: evolutionary-research-synthesis
description: >
  Use this skill when the user asks for a structured, evidence-grounded
  evolutionary analysis of any technology, product, scientific concept, system,
  method, industry practice, architecture, or design pattern. Trigger especially
  when the user wants a current concrete exemplar, prior development stages back
  to the earliest same-principle ancestor or representative functional concept,
  normalized performance metrics, a timeline and innovation-tagged graph, and
  one or more future successor concepts including both incremental and major leap
  innovations. Do not use for simple summaries, generic explanations, short
  opinion answers, or tasks where no historical or evolutionary comparison is
  requested.
---

# Evolutionary Research Synthesis Skill

## Purpose

Produce a rigorous, evidence-grounded evolutionary analysis for any user-selected topic.

This skill reconstructs how a present-day exemplar evolved from earlier stages, identifies the missing or newly added mechanisms at each stage, normalizes performance across time, and then proposes two successor concepts:

1. an **incremental successor** that plausibly improves the current best exemplar;
2. a **major leap successor** whose innovation significance is comparable to a large previous jump in the timeline.

The output should read like a structured research report, not a casual explanation.

---

## Input Contract

The user may provide any topic, for example:

- gasoline engine
- transformer architecture
- battery chemistry
- cloud-native application platform
- industrial PLC telemetry
- database engine
- photovoltaic inverter
- 3D printer
- AI coding agent
- water-meter telemetry system
- robotic actuator

The user may also specify:

- a concrete current exemplar to start from;
- a region or market;
- a time horizon;
- whether to include graphs or files;
- desired metrics;
- output language;
- report length.

If the user does not specify these, infer reasonable defaults and clearly label assumptions.

---

## Core Workflow

### 1. Frame the Research Question

Restate the topic as an evolutionary chain:

> I will analyze the evolution of [topic] by starting from a concrete current exemplar, tracing representative predecessor stages back to the earliest same-principle ancestor, normalizing key metrics, and then proposing incremental and leap successors.

Define the **same-principle boundary**:

- What counts as the same lineage?
- What is excluded as a different principle?
- Which early ancestors are ambiguous?
- Whether to include representative functional concepts, prototypes, or non-commercial designs.

Examples:

For a gasoline engine, distinguish:

- first internal-combustion piston engine;
- first compressed-charge four-stroke engine;
- first automotive gasoline engine.

For software or cloud topics, distinguish:

- same user-facing function;
- same architectural principle;
- same execution model;
- same abstraction boundary.

---

### 2. Research the Current Best Exemplar

Select a **specific modern concrete exemplar**, preferably:

- a real product, implementation, standard, paper, or architecture;
- well documented;
- high performing;
- representative of the current state of the art;
- not merely the most famous item.

For the current exemplar, capture:

- name and year or current status;
- creator, vendor, project, or research group;
- architecture or working principle;
- key performance metrics;
- physical or logical size and complexity;
- efficiency;
- reliability and safety constraints;
- cost or operational burden when available;
- limitations and bottlenecks.

Use up-to-date sources when the topic may have changed recently. Prefer primary sources such as official documentation, datasheets, standards, peer-reviewed papers, patents, vendor engineering notes, and reputable technical reviews.

---

### 3. Build the Backward Evolutionary Timeline

Trace backwards from the current exemplar to earlier development stages.

Use 6 to 12 stages by default.

Each stage may be either:

- a real product;
- a prototype;
- a research milestone;
- a standard;
- a representative functional concept;
- a reconstructed conceptual stage if a real product is unavailable.

For each stage, include:

- year or approximate period;
- representative artifact or concept;
- what it could already do;
- what it still lacked;
- key innovation introduced;
- key bottleneck solved;
- key bottleneck remaining;
- available metrics;
- confidence level.

Important: Do not force all stages to be commercial products. It is acceptable to include a functional concept missing a key component if that concept represents a genuine evolutionary stage.

---

### 4. Define Performance Metrics

Create a metric set appropriate to the topic.

Use two types of metrics.

#### A. Native metrics

Examples by domain:

- Engine: power, weight, displacement, thermal efficiency, noise, waste heat.
- Battery: Wh/kg, Wh/L, cycle life, C-rate, cost/kWh, safety.
- Transformer model: parameter count, context length, training cost, inference cost, benchmark score, latency.
- Cloud architecture: request latency, throughput, availability, recovery time, deployment frequency, operational cost.
- Industrial telemetry: battery life, transmission range, packet loss, installation cost, ingress protection, sampling rate.
- Database: query latency, write throughput, storage overhead, consistency guarantees, horizontal scalability.

#### B. Normalized metrics

Always add at least 3 normalized metrics, such as:

- performance per unit size;
- performance per unit mass;
- performance per unit cost;
- efficiency index;
- heat or waste index;
- complexity index;
- reliability index;
- autonomy index;
- maintainability index.

Normalize relative to the current exemplar unless another baseline is more useful.

Example:

- current exemplar equals 1.0;
- earlier stages show how far below or above they are;
- proposed successors show estimated improvement.

Clearly mark each metric as:

- measured value;
- estimated value;
- inferred value;
- unknown or unavailable.

---

### 5. Handle Uncertainty Correctly

For each metric, label data confidence:

- **High**: sourced from datasheet, paper, standard, or official documentation.
- **Medium**: reputable secondary source or cross-checked estimate.
- **Low**: engineering estimate, analogy, reconstructed concept, or incomplete historical data.

Never hide uncertainty.

Use ranges when exact values are unavailable.

Avoid false precision:

- prefer "approximately 150 to 170 kg" over "162.37 kg" unless sourced;
- prefer "order-of-magnitude improvement" when exact historical metrics are unreliable.

---

### 6. Synthesize Evolutionary Patterns

After the timeline, identify the major jumps.

For each jump, explain:

- what changed technically;
- why the change mattered;
- which constraint was removed;
- which new constraint appeared;
- how metrics changed;
- whether the improvement was mechanical, material, informational, architectural, economic, or regulatory.

Classify innovation types:

- **Mechanism innovation**: new working principle.
- **Architecture innovation**: rearranged components or boundaries.
- **Control innovation**: feedback, software, regulation, automation.
- **Material innovation**: new materials or manufacturing.
- **Energy/process innovation**: improved conversion, storage, flow, or thermodynamics.
- **System integration innovation**: product becomes usable because surrounding systems mature.
- **Economic innovation**: cost, mass production, distribution, business model.
- **Compliance/safety innovation**: standardization, regulation, reliability.

---

### 7. Create the Incremental Successor

Design a plausible next-generation version of the current best exemplar.

This successor must:

- preserve the core lineage;
- improve known bottlenecks;
- be technically plausible;
- avoid fantasy claims;
- state assumptions;
- include estimated metrics;
- include risks and trade-offs.

Output fields:

- name or concept label;
- core idea;
- architecture;
- key innovations;
- estimated performance;
- improvement versus current exemplar;
- expected constraints;
- readiness level;
- likely first application.

Use conservative estimates.

---

### 8. Create the Major Leap Successor

Design a more radical successor whose significance is comparable to a major historical jump found in the timeline.

This successor should:

- change at least one deep layer of the system:
  - mechanism;
  - architecture;
  - control paradigm;
  - material or process;
  - abstraction boundary;
  - energy conversion principle;
  - production or distribution model.
- deliver a step-change in normalized performance;
- still remain conceptually linked to the same problem domain;
- clearly explain why it qualifies as a leap.

Output fields:

- name or concept label;
- new principle;
- what it replaces;
- why it is analogous to a historical jump;
- estimated performance;
- normalized comparison;
- unresolved engineering barriers;
- plausible path to adoption.

Be explicit when the concept is speculative.

---

### 9. Produce Tables

Always include these tables unless the user asks for a shorter answer.

#### Table A: Evolutionary Timeline

Columns:

- Year or period
- Stage or representative
- Core principle
- Key innovation
- Missing component or limitation
- Key metrics
- Confidence

#### Table B: Normalized Performance

Columns:

- Stage
- Native performance metric 1
- Native performance metric 2
- Native performance metric 3
- Normalized performance index
- Efficiency or waste index
- Complexity or maintainability index
- Notes

#### Table C: Innovation Jumps

Columns:

- From to To
- Innovation
- Type
- Constraint removed
- Metric effect
- New trade-off

#### Table D: Successor Concepts

Columns:

- Concept
- Incremental or leap
- Principle
- Estimated performance
- Improvement versus current exemplar
- Main risks
- Readiness level

---

### 10. Produce a Graph When Possible

When enough numerical data exists, create at least one graph:

- x-axis: year or stage;
- y-axis: normalized performance or key metric;
- label points with key innovations;
- use log scale when improvements span orders of magnitude;
- annotate major jumps.

For non-numeric topics, use a structured maturity map instead:

- x-axis: time;
- y-axis: capability or abstraction level;
- labels: key architecture or control innovations.

If executing in an environment with Python, generate a PNG or SVG and save it. Use clear labels and avoid decorative clutter.

---

## Report Structure

Use this report format:

```markdown
# Evolutionary Analysis: [Topic]

## 1. Executive Summary

- current exemplar
- earliest ancestor
- biggest historical jumps
- incremental successor
- major leap successor
- most important uncertainty

## 2. Scope and Same-Principle Boundary

Define what is included and excluded.

## 3. Methodology and Sources

Explain:

- source priority
- metric selection
- normalization method
- uncertainty labels

## 4. Current Best Exemplar

Detailed technical profile.

## 5. Backward Timeline

Table and short narrative.

## 6. Normalized Performance Comparison

Table and explanation.

## 7. Innovation Jump Analysis

Identify major jumps and why they mattered.

## 8. Incremental Successor Concept

Plausible next-generation design.

## 9. Major Leap Successor Concept

Radical next-stage design.

## 10. Graph or Timeline Visualization

Include or link generated graph.

## 11. Critical Review

Explain weaknesses, assumptions, missing data, and alternative interpretations.

## 12. Conclusion

Summarize the evolutionary logic in a compact, decision-useful way.
