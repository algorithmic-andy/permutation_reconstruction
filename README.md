# Permutation Reasoning via Invariant-Guided World Models

A modular research framework for studying combinatorial reasoning, latent world modeling, and invariant-guided search in synthetic permutation environments.

This repository investigates whether predictive latent representations and structured planners can reason over hidden matrix permutations using only invariant-based observations.

The framework combines:

* invariant-guided search
* graph-based swap reasoning
* differentiable permutation flow models
* JEPA-style latent predictive architectures
* automated evaluation and reporting
* scaling and failure analysis tooling

The broader goal is to study how learned systems reason over structured combinatorial state spaces under constrained information.

---

# Core Research Themes

* combinatorial reasoning
* latent world modeling
* predictive representation learning
* invariant-guided search
* structured planning
* reasoning under uncertainty
* representation geometry
* trajectory optimization

---

# Current Capabilities

The repository currently supports:

* invariant-guided permutation environments
* swap-based combinatorial planners
* JEPA latent predictive models
* differentiable flow-based permutation models
* scaling experiments
* latent PCA analysis
* failure analysis
* multi-seed evaluation
* automated markdown reporting
* trajectory serialization
* runtime benchmarking
* hyperparameter sweeps
* statistical comparison tooling

---

# Research Motivation

Modern AI systems perform extremely well on many perception and prediction tasks, yet robust combinatorial reasoning remains difficult.

This project explores whether latent predictive world models can reason over structured permutation spaces using only invariant-based observations.

The environment is intentionally simple:

* shuffled integer matrices
* invariant constraints
* swap-based actions

Despite this simplicity, the underlying reconstruction problem becomes highly combinatorial as matrix size increases.

The broader goal is to study:

* latent reasoning
* combinatorial planning
* abstraction formation
* predictive world modeling

within a controlled experimental setting.

---

# Research Philosophy

This repository emphasizes:

* controlled synthetic environments
* interpretable reasoning dynamics
* modular experimentation
* lightweight reproducibility
* scalable research infrastructure

rather than large-scale compute optimization.

The goal is not only reconstruction accuracy, but understanding how reasoning systems behave under constrained structured information.

---

# Quickstart

## Installation

Clone the repository:

```bash
git clone <repo-url>
cd permutation-reasoning
```

Create environment:

```bash
python -m venv venv
```

Activate environment:

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -e .
```

Run tests:

```bash
pytest tests/
```

---

# Running Experiments

## Run Basic Experiment

```bash
python scripts/run_experiment.py
```

---

## Run Hyperparameter Sweep

```bash
python scripts/run_sweep.py
```

---

## Run Multi-Seed Evaluation

```bash
python scripts/run_multiseed_sweep.py
```

---

## Run Runtime Benchmark

```bash
python scripts/runtime_benchmark.py
```

---

## Run Full Evaluation Pipeline

```bash
python scripts/full_evaluation.py
```

---

# Generated Research Artifacts

Experiments automatically generate research artifacts including:

* benchmark summaries
* scaling curves
* latent PCA visualizations
* planner comparison plots
* convergence analysis
* runtime benchmarks
* failure distributions
* serialized trajectories
* markdown analysis reports
* statistical summaries

Generated outputs are stored inside experiment artifact directories.

Example artifact structure:

```text
artifacts/
└── experiment_name/
    ├── benchmark.json
    ├── analysis_report.md
    ├── latent_pca.png
    ├── scaling_curve.png
    ├── planner_comparison.png
    ├── failure_distribution.png
    └── trajectories/
```

---

# Example Generated Outputs

The framework supports automated generation of:

* latent representation geometry plots
* scaling analysis figures
* failure visualizations
* trajectory convergence plots
* markdown research reports
* planner comparison summaries

These artifacts are designed to support reproducible experimental analysis and lightweight research workflows.

---

# System Architecture

The repository is organized into modular research components:

* `dynamics/`
  permutation environments, invariants, transitions, search

* `models/`
  JEPA models, flow models, graph reasoning architectures

* `planners/`
  combinatorial search and action-selection policies

* `evaluation/`
  benchmarking, plotting, reporting, failure analysis, latent analysis

* `experiments/`
  sweeps, aggregation, serialization, runtime analysis

* `training/`
  training infrastructure and optimization

* `configs/`
  reproducible experiment configuration system

See:

* `docs/architecture.md`
* `docs/experiments.md`
* `docs/future_work.md`

---

# Experimental Methodology

## Environment

Each environment consists of:

* an `n x n` matrix
* values from `1` to `n²`
* randomly permuted target states

Agents observe invariant statistics derived from hidden target permutations.

---

## Invariant Observations

Current invariant observations include:

* row sums
* column sums
* row log-products
* column log-products

These invariants provide structured but indirect information about target configurations.

---

## Actions

Agents interact through swap operations between matrix positions.

This creates a combinatorial search problem over permutation space.

---

## Evaluation Metrics

Primary evaluation metrics include:

* mean hamming distance
* convergence rate
* trajectory length
* runtime scaling
* failure statistics

Lower hamming distance indicates more accurate reconstruction.

---

## Reproducibility

Experiments support:

* fixed random seeds
* multi-seed evaluation
* serialized outputs
* config-driven execution
* automated reporting

to improve scientific reproducibility and experimental consistency.

---

# Implemented Model Families

## Graph Swap Reasoning

Graph-style swap planners predict local permutation edits using relational reasoning dynamics.

Focus:

* explicit combinatorial search
* local optimization
* swap trajectory improvement

---

## Permutation Flow Models

Differentiable transport-style models predict soft permutation structure using Sinkhorn normalization.

Focus:

* permutation transport
* differentiable assignment structure
* global reasoning dynamics

---

## JEPA Latent Predictive Models

JEPA-style architectures learn latent predictive representations over permutation trajectories.

Focus:

* latent world modeling
* predictive representation learning
* trajectory reasoning
* energy-based latent consistency

---

# Evaluation and Analysis

The repository includes extensive analysis tooling for studying reasoning behavior.

Supported analysis includes:

* trajectory analysis
* convergence analysis
* scaling analysis
* latent PCA visualization
* failure diagnostics
* planner comparison
* runtime benchmarking
* statistical aggregation

The framework emphasizes interpretability and reasoning dynamics rather than only final reconstruction accuracy.

---

# Scaling Analysis

Scaling experiments study how reasoning difficulty changes as matrix dimensionality increases.

Scaling evaluations include:

* reconstruction difficulty
* convergence behavior
* runtime growth
* planner robustness
* failure frequency

Permutation complexity grows factorially with matrix size, making exact reconstruction increasingly difficult.

The repository intentionally emphasizes:

* lightweight reproducible experiments
* scaling trends
* reasoning behavior

rather than large-scale compute optimization.

---

# Failure Analysis

The framework includes tooling for analyzing difficult reconstruction failures.

Failure cases track:

* initial states
* target permutations
* final reconstructed states
* hamming distance
* trajectory length

This enables investigation of:

* unstable search trajectories
* local minima
* ambiguous invariant structures
* scaling bottlenecks

Understanding failure modes is important because exact combinatorial reconstruction rapidly becomes difficult as dimensionality increases.

---

# Latent World Modeling

The JEPA variants implemented in this repository investigate whether predictive latent representations can support combinatorial reasoning.

Rather than directly reconstructing permutations, these models learn:

* latent transition structure
* predictive trajectory representations
* invariant-aware embeddings

This framing is inspired by broader research directions in:

* predictive world models
* latent planning
* representation learning
* self-supervised reasoning systems

The goal is not merely prediction accuracy, but understanding how latent predictive structure may support reasoning over discrete combinatorial environments.

---

# Key Research Insights

## Invariants Provide Strong Structural Constraints

Even simple invariant statistics dramatically constrain permutation search space structure.

This suggests that reasoning systems may benefit from abstract constraint representations rather than direct reconstruction objectives alone.

---

## Exact Reconstruction Is Highly Combinatorial

As matrix dimensionality grows, exact reconstruction rapidly becomes difficult.

However, models often learn meaningful partial structure and substantially reduce hamming distance even when exact recovery fails.

---

## Latent Prediction Encourages Structured Representations

JEPA-style objectives frequently learn smoother latent transition geometry than direct action prediction objectives.

This may indicate that predictive latent modeling encourages more globally coherent reasoning representations.

---

## Search and Representation Learning Are Complementary

Explicit search-based planners and latent predictive models exhibit different strengths:

* search improves local optimization
* latent models improve global structure learning

Future systems may benefit from combining both approaches.

---

# Research Infrastructure

The repository includes modular experimentation infrastructure for reproducible research workflows.

Supported infrastructure includes:

* hyperparameter sweeps
* multi-seed evaluation
* runtime benchmarking
* trajectory serialization
* automated markdown reporting
* artifact generation
* statistical aggregation
* scaling evaluation

The goal is to support lightweight but extensible combinatorial reasoning research.

---

# Research Questions

This repository investigates several questions related to combinatorial reasoning and latent world modeling.

## Core Questions

1. Can latent predictive models reason over combinatorial permutation spaces?

2. How well can invariant-guided planners reconstruct hidden matrix permutations?

3. How do JEPA-style latent objectives compare against explicit search-based methods?

4. What representations emerge during invariant-based reasoning?

5. How does reasoning performance scale with matrix dimensionality?

---

## Comparative Questions

The repository compares multiple reasoning paradigms:

* graph-based swap reasoning
* permutation flow transport models
* latent predictive JEPA systems

The goal is not only reconstruction accuracy, but also:

* convergence behavior
* planning efficiency
* latent structure formation
* scaling properties

---

# Future Work

Several important research directions remain open.

## Partial and Noisy Invariant Reasoning

Current experiments assume access to complete invariant information.

Future work will investigate:

* partial invariant observations
* noisy measurements
* probabilistic reconstruction
* uncertainty-aware planning

---

## Hierarchical Planning

Current planners primarily operate through local swap actions.

Future systems may explore:

* hierarchical search
* latent planning
* tree-based reasoning
* coarse-to-fine reconstruction

---

## Representation Geometry

Additional work is needed to better understand:

* latent transition structure
* invariant compression
* geometric organization of reasoning trajectories

---

# Limitations

This project is intentionally designed as a lightweight research framework rather than a large-scale production system.

Current limitations include:

* relatively small matrix sizes
* synthetic environments
* limited planner complexity
* lightweight compute budgets
* full invariant observability assumptions

The framework prioritizes:

* modularity
* reasoning analysis
* reproducibility
* experimental clarity

over large-scale optimization.

---

# Repository Structure

```text
src/permutation_reasoning/
├── data/
├── dynamics/
├── evaluation/
├── experiments/
├── models/
├── planners/
├── training/
└── utils/

configs/
├── data/
├── experiment/
├── model/
└── train/

scripts/
├── run_experiment.py
├── run_sweep.py
├── run_multiseed_sweep.py
├── run_scaling.py
├── runtime_benchmark.py
├── compare_models.py
└── full_evaluation.py

docs/
tests/
artifacts/
```

---

# Current Status

The repository is actively under development.

Current focus areas include:

* latent planning
* noisy invariant reasoning
* hierarchical search
* scaling analysis
* combinatorial world modeling

The framework is intentionally modular to support future experimentation and research extensions.

---

# Citation

If you use this repository in future research, please cite:

```bibtex
@misc{permutation_reasoning,
  title={Permutation Reasoning via Invariant-Guided World Models},
  author={Josh King},
  year={2026},
}
```

---

# Conclusion

This repository explores combinatorial reasoning through invariant-guided permutation reconstruction and latent predictive world modeling.

The framework combines:

* modular experimentation
* structured search
* predictive latent modeling
* scaling analysis
* failure analysis
* automated research reporting

within a unified research infrastructure.

While the environment is intentionally synthetic, the broader motivation is to study how learned systems reason over structured state spaces under constrained information.

The long-term goal is to better understand:

* latent reasoning
* abstraction
* predictive planning
* combinatorial world modeling

in scalable AI systems.
