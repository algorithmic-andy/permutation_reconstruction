# Predicting Combinatorial Reconstruction Difficulty

## Overview

This project investigates whether the computational difficulty of reconstructing a permutation matrix from partial invariant information can be predicted using conventional regression methods.

For a fixed natural number (n), consider a permutation of the integers

$$
1,2,\ldots,n^2
$$

reshaped into an ($n\times n$) matrix. Each matrix has several invariant statistics, including the sum and product of each row and column, as well as the corresponding statistics for the main diagonal and anti-diagonal.

A subset of these invariants is observed, producing a **Partial Observation Set**. The resulting constraints define a combinatorial reconstruction problem, which is solved using Google's OR-Tools CP-SAT constraint solver.

The central question is:

> **Can the computational difficulty of reconstructing a permutation matrix from partial invariant information be predicted from the observed invariants?**

Three regression methods are compared:

* Linear regression
* Ridge regression
* Regression tree

The project is divided into two independent phases:

1. **Dataset generation** — generates permutation reconstruction instances and records CP-SAT solver statistics.
2. **Statistical modeling** — converts the partial observations into feature vectors and compares the predictive performance of the three regression methods.

---

## Mathematical Problem

Let

$$
A\in\mathbb{N}^{n\times n}
$$

be a matrix containing each integer from (1) through (n^2) exactly once.

For this project, the dimension is fixed at

$$
n=5,
$$

so each matrix contains the integers ($1,\ldots,25$).

For each matrix, the following invariants can be computed:

### Rows

For each row (i),

$$
S_i^{(r)} = \sum_{j=1}^{n} A_{ij},
$$

and

$$
P_i^{(r)} = \prod_{j=1}^{n} A_{ij}.
$$

### Columns

For each column (j),

$$
S_j^{(c)} = \sum_{i=1}^{n} A_{ij},
$$

and

$$
P_j^{(c)} = \prod_{i=1}^{n} A_{ij}.
$$

### Diagonal

The sum and product of the main diagonal are also recorded.

### Anti-diagonal

The sum and product of the anti-diagonal are recorded as well.

Thus, there are

$$
4n+4
$$

potential invariant statistics. For (n=5), this gives

$$
4(5)+4=24
$$

possible features.

---

## Partial Observations

For each generated matrix, every invariant is independently observed with probability

$$
p=0.5.
$$

Thus, each instance contains a different Partial Observation Set.

The observed constraints are passed through a closure procedure before reconstruction. The resulting constraint system is given to the CP-SAT solver.

This creates instances with varying levels of information and, consequently, varying levels of computational difficulty.

---

## Dataset Generation

The first phase generates the computational dataset.

For each instance:

1. A permutation of ($1,\ldots,n^2$) is generated.
2. The permutation is reshaped into an ($n\times n$) matrix.
3. The matrix is canonicalized to remove equivalent configurations.
4. All invariant statistics are computed.
5. Each invariant is independently sampled with probability ($p=0.5$).
6. Closure is applied to the observed constraints.
7. CP-SAT attempts to reconstruct the matrix.
8. Solver statistics are recorded.

The primary solver responses are:

* **Branches** — number of search branches explored.
* **Conflicts** — number of conflicts encountered.
* **Propagations** — number of constraint propagations.
* **Solutions** — number of solutions found.

Runtime is not used as a primary response because it depends heavily on the hardware and environment on which the solver is executed.

The number of solutions is capped at 50. If more than 50 solutions are found, the recorded value is therefore 50.

---

## Regression Response

The primary response variable is solver search difficulty, measured by the number of branches.

The raw branch count is strongly right-skewed, so the response is transformed as

$$
Y=\log(1+\text{branches}).
$$

The logarithmic transformation reduces the influence of extremely difficult instances and produces a more suitable response for conventional regression methods.

Conflicts and propagations are retained as potential secondary responses and diagnostics.

---

## Feature Representation

Each Partial Observation Set is converted into a fixed-length vector of 24 features.

The feature ordering is:

| Features              | Number |
| --------------------- | -----: |
| Row sums              |      5 |
| Row products          |      5 |
| Column sums           |      5 |
| Column products       |      5 |
| Diagonal sum          |      1 |
| Diagonal product      |      1 |
| Anti-diagonal sum     |      1 |
| Anti-diagonal product |      1 |
| **Total**             | **24** |

### Product transformation

Raw product values can be substantially larger than the corresponding sums. Therefore, product observations are logarithmically transformed before entering the regression models:

$$
P \longrightarrow \log(P).
$$

The `"product"` identifier is retained in the observation representation; the logarithmic transformation occurs when constructing the regression feature vector.

Sum observations are left unchanged.

### Missing observations

An unobserved invariant is represented by zero.

This is possible because neither a valid sum nor a valid product in this problem can naturally equal zero.

Consequently, every Partial Observation Set can be represented using the same 24-dimensional feature vector, regardless of how many invariants were observed.

A potential limitation is that the regression model must infer that zero represents an unobserved statistic rather than an observed value. Adding explicit observation indicators is therefore an important direction for future work.

---

## Regression Methods

Three regression methods are compared.

### 1. Linear Regression

Ordinary linear regression provides the simplest and most interpretable baseline:


$E[Y\mid X]$
==========
$$
\beta_0+\sum_{j=1}^{24}\beta_jX_j.
$$

This determines whether reconstruction difficulty can be reasonably approximated using an additive linear relationship between the observed invariants and solver difficulty.

### 2. Ridge Regression

Ridge regression adds an (L_2) penalty to the linear regression objective:


$\hat{\beta}$
===========

$$
\arg\min_\beta
\left[
\sum_i(y_i-x_i^\top\beta)^2
+
\lambda\sum_j\beta_j^2
\right].
$$

Ridge is motivated by the substantial dependence expected among the invariant features.

For example, row and column statistics are constrained by the fact that the matrix contains every integer from (1) through (n^2) exactly once. Symmetries such as matrix transposition and row or column permutations also create relationships between different feature configurations.

### 3. Regression Tree

A regression tree is included to investigate whether nonlinear relationships and interactions among invariants improve predictive performance.

Unlike linear regression and Ridge, a tree can represent relationships in which the effect of one invariant depends on the value of another.

This is particularly relevant because the underlying reconstruction problem is a constrained combinatorial problem.

---

## Evaluation

The models are evaluated using common predictive metrics:

### Mean Absolute Error

$$
MAE =
\frac{1}{N}
\sum_{i=1}^{N}
|y_i-\hat y_i|.
$$

MAE measures the average absolute prediction error and is relatively easy to interpret.

### Root Mean Squared Error

$$
RMSE =
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat y_i)^2
}.
$$

RMSE gives greater weight to large prediction errors than MAE.

### Coefficient of Determination

$$
R^2
===

1-
\frac{
\sum_i(y_i-\hat y_i)^2
}{
\sum_i(y_i-\bar y)^2
}.
$$

(R^2) is the primary metric because it measures the proportion of variation in solver difficulty explained by the model relative to a mean-response baseline.

---

## Results

Using the current experimental configuration, the following results were obtained:

| Model  |        MAE |       RMSE |      (R^2) |
| ------ | ---------: | ---------: | ---------: |
| Linear |     1.4025 |     2.0091 | **0.5084** |
| Ridge  | **1.3643** | **2.0124** |     0.5077 |
| Tree   |     1.7868 |     2.6347 |     0.1569 |

The linear and Ridge models perform almost identically. The small difference suggests that regularization provides little additional predictive benefit in this experiment, despite the expected correlation among the invariant features.

The regression tree performs substantially worse, with

$$
R^2=0.1569.
$$

This indicates that the particular nonlinear structure captured by the regression tree does not provide a better approximation to solver difficulty than the global linear models under the current feature representation and experimental design.

---

## Interpretation

The results demonstrate that the invariant statistics contain meaningful information about computational difficulty. The linear models explain approximately half of the variation in the log-transformed number of search branches.

However, the results also suggest that conventional tabular regression models do not fully capture the structure of the underlying combinatorial problem.

The underlying population is extremely large. For (n=5), there are

$$
25!
$$

possible permutation matrices, approximately

$$
1.55\times10^{25}.
$$

The 10,000 generated instances therefore represent only a tiny fraction of the possible permutation matrices.

This does **not** imply that the reported test performance is necessarily biased in a particular direction, but it does limit the extent to which the results can be interpreted as describing the entire combinatorial population. The sampled instances provide evidence about the data-generating process represented by the experiment, rather than exhaustive coverage of the underlying permutation space.

The substantially poorer performance of the regression tree may also reflect the difficulty of learning a structured combinatorial landscape using axis-aligned partitions from a finite sample. The observed invariant configurations occupy a highly constrained subset of the nominal 24-dimensional feature space, and the relationships between the features are determined by the underlying permutation structure.

---

## Future Work

Several extensions could investigate whether explicitly modeling the combinatorial structure improves predictive performance.

### Graph Neural Networks

A graph neural network (GNN) is a natural candidate for future work because the problem contains an explicit relational structure.

One possible representation would use invariant statistics as nodes and matrix entries as connected nodes, with edges representing the relationships between statistics and their corresponding matrix elements.

Alternatively, matrix elements could be represented as nodes, with invariant statistics defining relationships between them.

These representations could allow a model to exploit structural relationships that are lost when the Partial Observation Set is flattened into a fixed-length feature vector.

### Observation Indicators

The current representation uses zero to indicate that an invariant was not observed.

A future experiment could add 24 binary indicators:

$$
M_j=
\begin{cases}
1 & \text{if invariant }j\text{ is observed},\
0 & \text{otherwise}.
\end{cases}
$$

The resulting feature vector would contain 48 variables:

$$
(X_1,\ldots,X_{24},M_1,\ldots,M_{24}).
$$

This would allow the regression models to distinguish explicitly between an unobserved statistic and an observed numerical value.

### Additional Responses

Future work could investigate whether conflicts, propagations, or other solver statistics provide complementary measures of computational difficulty.

Composite difficulty measures and ratios between solver statistics could also be explored, although these would require additional justification.

### Additional Matrix Dimensions

The current experiment fixes

$$
n=5.
$$

Future studies could investigate how predictive performance changes as (n) increases or decreases.

This would also provide a direct way to study whether the difficulty of learning the computational landscape increases with the size of the underlying combinatorial search space.

### Structured Machine Learning

Other models that explicitly incorporate interactions or combinatorial structure could also be considered, including more sophisticated tree ensembles, neural networks, and graph-based models.

---

## Project Structure

The project is organized into two independent phases.

```text
project/
│
├── data/
│   └── generate_dataset.py
│
├── models/
│   ├── pipeline.py
│   ├── metrics.py
│   ├── compare.py
│   └── ...
│
├── utils/
│   ├── permutation.py
│   ├── canonical.py
│   ├── observations.py
│   ├── closure.py
│   └── features.py
│
├── solver/
│   └── reconstruct.py
│
├── core/
│   └── types.py
│
├── datasets/
│   └── dataset.pkl
│
├── config.py
├── requirements.txt
└── README.md
```

### Phase 1: Dataset Generation

The dataset generation phase is responsible for:

* generating permutation matrices,
* computing invariant statistics,
* sampling Partial Observation Sets,
* applying observation closure,
* solving reconstruction problems with CP-SAT,
* recording solver statistics,
* and saving the resulting dataset.

The output is stored independently of the regression analysis so that the computationally expensive dataset does not need to be regenerated when experimenting with different statistical models.

### Phase 2: Statistical Modeling

The modeling phase is responsible for:

* loading the generated dataset,
* constructing feature vectors,
* extracting response variables,
* splitting the data into training and test sets,
* fitting regression models,
* evaluating predictions,
* and comparing model performance.

This separation allows the statistical analysis to be modified without rerunning the computational dataset generation.

---

## Reproducibility

The experiment uses fixed random seeds for reproducibility.

Key experimental parameters are centralized in `config.py`, including:

* matrix dimension (n),
* number of generated instances,
* observation probability,
* random seed.

The primary experiment uses:

```text
n = 5
number of instances = 10,000
observation probability = 0.5
```

The dataset generation phase should be run before the modeling phase.

---

## Dependencies

The project is implemented in Python and primarily uses:

* NumPy
* pandas
* scikit-learn
* Matplotlib
* Google OR-Tools

See `requirements.txt` for the complete dependency specification.

---

## Limitations

Several limitations should be considered when interpreting the results.

1. **Finite sampling of a massive combinatorial population.**
   The experiment uses 10,000 instances from an enormously larger space of possible permutation matrices.

2. **Fixed dimension.**
   Only (n=5) is considered, so the conclusions may not generalize to other matrix dimensions.

3. **Fixed observation probability.**
   The primary experiment uses (p=0.5). Different observation probabilities may produce different relationships between invariant information and reconstruction difficulty.

4. **Missingness representation.**
   Unobserved invariants are encoded as zero rather than using explicit missingness indicators.

5. **Censored solution count.**
   The number of solutions is capped at 50 and therefore does not represent the exact number of solutions for instances exceeding this threshold.

6. **Solver-specific responses.**
   The response variables are statistics produced by CP-SAT and may depend on solver implementation and configuration.

7. **Model class limitations.**
   The three regression models considered here do not explicitly encode the combinatorial structure of the permutation reconstruction problem.

---

## Summary

This project explores a connection between **combinatorial constraint solving and statistical prediction**.

The key finding is that partial invariant information contains substantial predictive information about CP-SAT reconstruction difficulty, with linear and Ridge regression explaining approximately half of the variation in log-transformed search branches. However, the substantially weaker performance of the regression tree suggests that simply introducing a more flexible tabular model does not necessarily capture the underlying combinatorial structure.

The results motivate future investigation of models that incorporate the structure of the permutation reconstruction problem directly, particularly graph-based representations and neural architectures designed for relational data.
