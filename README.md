# Permutation Reasoning

A modular research framework for latent combinatorial search under invariant constraints.

---

## Overview

This project investigates whether neural latent world models can recover hidden combinatorial structure from compressed invariant observations.

Given:

* an `n x n` matrix containing a permutation of integers `1..n²`,
* and invariant observations derived from that hidden arrangement,

models must iteratively refine candidate permutations toward the latent target configuration.

Unlike standard supervised reconstruction tasks, the focus is not solely exact recovery, but:

* iterative search dynamics,
* latent planning,
* structured representation learning,
* and progressive reduction in combinatorial inconsistency.

The framework is designed as a controlled synthetic environment for studying:

* latent reasoning,
* predictive representation learning,
* permutation-space search,
* and structured world models.

---

# Problem Definition

Let:

* `X ∈ R^(n×n)` be a hidden permutation matrix containing integers `1..n²`
* `I(X)` denote a set of invariant statistics computed from `X`

The invariant operator currently includes:

* row sums,
* column sums,
* row log-products,
* column log-products.

The task is to learn:

```text
I(X) -> X
```

or more generally:

```text
candidate_state_t
    + invariant observations
    -> improved candidate_state_(t+1)
```

through iterative combinatorial search.

---

# Research Goals

This repository explores:

1. Latent representation learning over permutation spaces
2. Iterative refinement dynamics for combinatorial reasoning
3. JEPA-style predictive world models for structured search
4. Energy-based reasoning over invariant consistency
5. Comparison between graph, transport, and latent planning approaches

The project intentionally emphasizes:

* modularity,
* reproducibility,
* clean experiment design,
* and interpretable research structure.

---

# Core Research Questions

* Can latent predictive models learn meaningful search trajectories in permutation space?
* Do learned latent dynamics improve combinatorial reconstruction efficiency?
* How do graph-based, flow-based, and JEPA-based approaches differ in latent geometry?
* Can iterative latent planning reduce invariant inconsistency over time?
* How does reasoning performance scale with permutation complexity?

---

# Methods

The framework currently targets three model families.

## 1. Graph-Based Search Models

Graph neural architectures operating over pairwise permutation interactions.

Focus:

* relational reasoning,
* swap prediction,
* local combinatorial structure.

---

## 2. Permutation Flow Models

Transport-style models that learn soft transition operators over permutation states.

Focus:

* global assignment structure,
* soft permutation dynamics,
* Sinkhorn-based transport reasoning.

---

## 3. JEPA-Based Latent World Models

Joint embedding predictive architectures that learn latent transition dynamics.

Focus:

* predictive latent representations,
* iterative refinement,
* latent planning,
* EMA target encoders,
* trajectory consistency.

---

# Project Philosophy

This repository is intentionally designed as:

```text
A small but deeply coherent research framework.
```

The emphasis is on:

* modular abstractions,
* clean experimental interfaces,
* compositional model design,
* and reproducible research workflows.

The goal is not infrastructure scale.

The goal is:

* scientific clarity,
* architectural discipline,
* and meaningful experimentation.

---

# Repository Structure

```text
permutation-reasoning/
│
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── configs/
│   ├── data/
│   ├── model/
│   ├── train/
│   └── experiment/
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── benchmark.py
│   └── analyze_latents.py
│
├── src/
│   └── permutation_reasoning/
│       ├── dynamics/
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       └── utils/
│
├── tests/
├── outputs/
├── notebooks/
└── docs/
```

---

# Core Abstractions

The framework is built around a small set of semantic primitives.

## PermutationState

Represents a candidate permutation configuration.

## Invariants

Structured invariant observations derived from hidden target states.

## SwapAction

Atomic permutation transition operator.

## Transition

Single search step:

```text
state_t + action_t -> state_(t+1)
```

## Trajectory

Sequence of search transitions through permutation space.

---

# Evaluation

Primary evaluation metrics include:

* Exact reconstruction accuracy
* Hamming distance
* Invariant residual error
* Search efficiency
* Trajectory improvement
* Scaling behavior with increasing `n`

The project focuses heavily on:

* failure analysis,
* ambiguity structure,
* and latent representation geometry.

---

# Planned Features

* Modular JEPA framework
* EMA target encoders
* Iterative latent planners
* Beam-search refinement
* Curriculum scaling across `n`
* Latent trajectory analysis
* Energy-based consistency scoring
* Representation probing utilities
* Visualization and benchmarking tools

---

# Development Roadmap

## M0 — Research Architecture & System Design

* Problem framing
* Modular abstraction design
* Experiment specification

## M1 — Transition Dynamics & Dataset Foundation

* State representations
* Invariant computation
* Search dynamics
* Dataset generation

## M2 — Shared Neural Infrastructure

* Feature extraction
* Encoders
* Registries
* Model interfaces

## M3 — Baseline Models

* Graph search baseline
* Permutation flow baseline

## M4 — JEPA World Models

* Latent prediction
* EMA target networks
* Iterative refinement

## M5 — Evaluation & Analysis

* Benchmarking
* Scaling analysis
* Latent probing
* Failure analysis

---

# Status

Current phase:

```text
M1 — Transition Dynamics & Dataset Foundation
```

---

# Long-Term Vision

This project aims to serve as a compact but research-grade framework for studying:

```text
latent reasoning over combinatorial state spaces.
```

The broader motivation is understanding how predictive latent world models can support:

* structured reasoning,
* iterative search,
* and combinatorial generalization.

---

# License

MIT License
