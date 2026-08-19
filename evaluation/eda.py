from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt



def load_dataset(path="datasets/dataset.pkl"):
    """
    Load the generated dataset from disk.

    Parameters
    ----------
    path : str
        Path to the pickle file.

    Returns
    -------
    list
        Dataset as a list of dictionaries.
    """

    with open(path, "rb") as f:
        dataset = pickle.load(f)

    print(f"Loaded {len(dataset)} instances from {path}")

    return dataset


def dataset_summary(dataset):
    """
    Print a numerical summary of the dataset.
    """

    stats = [d["solver_stats"] for d in dataset]

    branches = np.array([s.branches for s in stats])
    conflicts = np.array([s.conflicts for s in stats])
    propagations = np.array([s.propagations for s in stats])
    runtimes = np.array([s.runtime for s in stats])
    num_solutions = np.array([s.num_solutions for s in stats])

    observation_counts = np.array([
        len(d["raw_observations"])
        for d in dataset
    ])

    closure_sizes = np.array([
        len(d["closed_observations"])
        for d in dataset
    ])

    closure_gain = closure_sizes - observation_counts

    solved = np.sum([s.solved for s in stats])

    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"Instances                : {len(dataset)}")
    print(f"Solved                   : {solved}")
    print(f"Unsolved                 : {len(dataset)-solved}")

    print()

    print(f"Average runtime          : {runtimes.mean():.6f} s")
    print(f"Median runtime           : {np.median(runtimes):.6f} s")

    print()

    print(f"Average branches         : {branches.mean():.2f}")
    print(f"Median branches          : {np.median(branches):.2f}")

    print()

    print(f"Average conflicts        : {conflicts.mean():.2f}")
    print(f"Median conflicts         : {np.median(conflicts):.2f}")

    print()

    print(f"Average propagations     : {propagations.mean():.2f}")
    print(f"Median propagations      : {np.median(propagations):.2f}")

    print()

    print(f"Average # solutions      : {num_solutions.mean():.2f}")
    print(f"Median # solutions       : {np.median(num_solutions):.2f}")

    print()

    print(f"Average observations     : {observation_counts.mean():.2f}")
    print(f"Median observations      : {np.median(observation_counts):.2f}")

    print()

    print(f"Average closure size     : {closure_sizes.mean():.2f}")
    print(f"Median closure size      : {np.median(closure_sizes):.2f}")

    print()

    print(f"Average closure gain     : {closure_gain.mean():.2f}")
    print(f"Median closure gain      : {np.median(closure_gain):.2f}")

    print("=" * 60)


def plot_solver_histograms(dataset):
    """
    Plot histograms of all solver statistics.
    """

    stats = [d["solver_stats"] for d in dataset]

    plots = [
        ("Branches",
         np.array([s.branches for s in stats])),

        ("log(1 + Branches)",
         np.log1p([s.branches for s in stats])),

        ("Conflicts",
         np.array([s.conflicts for s in stats])),

        ("log(1 + Conflicts)",
         np.log1p([s.conflicts for s in stats])),

        ("Propagations",
         np.array([s.propagations for s in stats])),

        ("Runtime (seconds)",
         np.array([s.runtime for s in stats])),

        ("Number of Solutions",
         np.array([s.num_solutions for s in stats])),

        ("log(1 + Number of Solutions)",
         np.log1p([s.num_solutions for s in stats])),
    ]

    for title, values in plots:

        plt.figure(figsize=(7,5))

        plt.hist(
            values,
            bins=40,
            edgecolor="black",
        )

        plt.title(title)
        plt.xlabel(title)
        plt.ylabel("Frequency")

        plt.tight_layout()

    plt.show()


def plot_dataset_histograms(dataset):
    """
    Plot histograms describing the observation sets.
    """

    observation_counts = np.array([
        len(d["raw_observations"])
        for d in dataset
    ])

    closure_sizes = np.array([
        len(d["closed_observations"])
        for d in dataset
    ])

    closure_gain = closure_sizes - observation_counts

    plots = [

        ("Observation Count",
         observation_counts),

        ("Closure Size",
         closure_sizes),

        ("Closure Gain",
         closure_gain),
    ]

    for title, values in plots:

        plt.figure(figsize=(7,5))

        plt.hist(
            values,
            bins=30,
            edgecolor="black",
        )

        plt.title(title)
        plt.xlabel(title)
        plt.ylabel("Frequency")

        plt.tight_layout()

    plt.show()


def plot_correlation_heatmap(dataset):
    """
    Plot a Pearson correlation heatmap between all major variables.
    """

    stats = [d["solver_stats"] for d in dataset]

    branches = np.array([s.branches for s in stats], dtype=float)
    conflicts = np.array([s.conflicts for s in stats], dtype=float)
    propagations = np.array([s.propagations for s in stats], dtype=float)
    runtime = np.array([s.runtime for s in stats], dtype=float)
    solutions = np.array([s.num_solutions for s in stats], dtype=float)

    observations = np.array([
        len(d["raw_observations"])
        for d in dataset
    ], dtype=float)

    closure = np.array([
        len(d["closed_observations"])
        for d in dataset
    ], dtype=float)

    gain = closure - observations

    data = np.vstack([
        np.log1p(branches),
        np.log1p(conflicts),
        np.log1p(propagations),
        runtime,
        np.log1p(solutions),
        observations,
        closure,
        gain,
    ])

    labels = [
        "log(Branches)",
        "log(Conflicts)",
        "log(Propagations)",
        "Runtime",
        "log(Solutions)",
        "Observations",
        "Closure",
        "Gain",
    ]

    corr = np.corrcoef(data)

    plt.figure(figsize=(8, 7))

    plt.imshow(corr, vmin=-1, vmax=1)

    plt.colorbar(label="Pearson Correlation")

    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(
                j,
                i,
                f"{corr[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.show()


def _scatter(x, y, xlabel, ylabel, title):
    """
    Publication-quality scatter plot with regression line.
    """

    r = np.corrcoef(x, y)[0, 1]

    plt.figure(figsize=(7, 5))

    plt.scatter(
        x,
        y,
        s=8,
        alpha=0.35,
    )

    m, b = np.polyfit(x, y, 1)

    xs = np.linspace(np.min(x), np.max(x), 200)

    plt.plot(xs, m * xs + b, linewidth=2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.title(f"{title}\nPearson r = {r:.3f}")

    plt.tight_layout()

    plt.show()


def plot_solver_relationships(dataset):
    """
    Visualize relationships between solver statistics.
    """

    stats = [d["solver_stats"] for d in dataset]

    branches = np.log1p(np.array([s.branches for s in stats]))
    conflicts = np.log1p(np.array([s.conflicts for s in stats]))
    propagations = np.log1p(np.array([s.propagations for s in stats]))
    runtime = np.array([s.runtime for s in stats])
    solutions = np.log1p(np.array([s.num_solutions for s in stats]))

    _scatter(
        runtime,
        branches,
        "Runtime (seconds)",
        "log(1 + Branches)",
        "Runtime vs Branches",
    )

    _scatter(
        conflicts,
        branches,
        "log(1 + Conflicts)",
        "log(1 + Branches)",
        "Conflicts vs Branches",
    )

    _scatter(
        propagations,
        branches,
        "log(1 + Propagations)",
        "log(1 + Branches)",
        "Propagations vs Branches",
    )

    _scatter(
        solutions,
        branches,
        "log(1 + Number of Solutions)",
        "log(1 + Branches)",
        "Solutions vs Branches",
    )


def plot_observation_relationships(dataset):
    """
    Relationships between observation structure and solver difficulty.
    """

    stats = [d["solver_stats"] for d in dataset]

    branches = np.log1p(np.array([s.branches for s in stats]))

    observations = np.array([
        len(d["raw_observations"])
        for d in dataset
    ], dtype=float)

    closure = np.array([
        len(d["closed_observations"])
        for d in dataset
    ], dtype=float)

    gain = closure - observations

    solutions = np.log1p(np.array([s.num_solutions for s in stats]))

    # --------------------------------------------------------
    # Observation count vs difficulty
    # --------------------------------------------------------

    _scatter(
        observations,
        branches,
        "Observation Count",
        "log(1 + Branches)",
        "Observations vs Branches",
    )

    # --------------------------------------------------------
    # Closure size vs difficulty
    # --------------------------------------------------------

    _scatter(
        closure,
        branches,
        "Closure Size",
        "log(1 + Branches)",
        "Closure Size vs Branches",
    )

    # --------------------------------------------------------
    # Closure gain vs difficulty
    # --------------------------------------------------------

    _scatter(
        gain,
        branches,
        "Closure Gain",
        "log(1 + Branches)",
        "Closure Gain vs Branches",
    )

    # --------------------------------------------------------
    # Ambiguity vs difficulty
    # --------------------------------------------------------

    _scatter(
        solutions,
        branches,
        "log(1 + Number of Solutions)",
        "log(1 + Branches)",
        "Solutions vs Branches",
    )

def plot_phase_transition(dataset):
    """
    Phase transition analysis:
    Probability of unique solution given observation count
    and closure size.
    """

    stats = [d["solver_stats"] for d in dataset]

    num_solutions = np.array([
        s.num_solutions for s in stats
    ], dtype=int)

    is_unique = (num_solutions == 1).astype(int)

    observations = np.array([
        len(d["raw_observations"])
        for d in dataset
    ], dtype=int)

    closure = np.array([
        len(d["closed_observations"])
        for d in dataset
    ], dtype=int)

    # ========================================================
    # Helper: compute empirical probability curve
    # ========================================================

    def compute_curve(x, y, bins=30):
        """
        Compute P(y=1 | x) via binning.
        """

        bins_edges = np.linspace(x.min(), x.max(), bins + 1)
        centers = 0.5 * (bins_edges[:-1] + bins_edges[1:])

        probs = []

        for i in range(bins):
            mask = (x >= bins_edges[i]) & (x < bins_edges[i + 1])

            if np.sum(mask) < 10:
                probs.append(np.nan)
            else:
                probs.append(np.mean(y[mask]))

        return centers, np.array(probs)

    # ========================================================
    # Observation count curve
    # ========================================================

    x1, p1 = compute_curve(observations, is_unique)

    plt.figure(figsize=(7, 5))
    plt.plot(x1, p1, marker="o")
    plt.title("P(Unique Solution | Observation Count)")
    plt.xlabel("Number of Observations")
    plt.ylabel("Probability of Unique Solution")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()

    plt.show()

    # ========================================================
    # Closure size curve
    # ========================================================

    x2, p2 = compute_curve(closure, is_unique)

    plt.figure(figsize=(7, 5))
    plt.plot(x2, p2, marker="o")
    plt.title("P(Unique Solution | Closure Size)")
    plt.xlabel("Closure Size")
    plt.ylabel("Probability of Unique Solution")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()

    plt.show()


def feature_audit(dataset):
    """
    Analyze predictive power of each of the 24 features.
    """

    stats = [d["solver_stats"] for d in dataset]

    target = np.log1p(np.array([s.branches for s in stats]))

    X = np.array([
        d["feature_vector"]
        for d in dataset
    ], dtype=float)

    n_features = X.shape[1]

    # --------------------------------------------------------
    # Masks for observed values
    # --------------------------------------------------------

    observed_mask = (X != 0)

    freq = observed_mask.mean(axis=0)
    mean_value = np.where(observed_mask, X, np.nan).mean(axis=0)

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    def corr(x, y):
        if np.std(x) == 0 or np.std(y) == 0:
            return np.nan
        return np.corrcoef(x, y)[0, 1]

    corr_target = np.array([
        corr(X[:, i], target)
        for i in range(n_features)
    ])

    # --------------------------------------------------------
    # Print summary table
    # --------------------------------------------------------

    print("=" * 70)
    print("FEATURE AUDIT (24-DIMENSIONAL VECTOR)")
    print("=" * 70)

    for i in range(n_features):
        print(
            f"Feature {i:02d} | "
            f"freq={freq[i]:.3f} | "
            f"mean={mean_value[i]:.3f} | "
            f"corr(branches)={corr_target[i]:.3f}"
        )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    plt.figure(figsize=(10, 4))
    plt.bar(range(n_features), corr_target)
    plt.title("Feature Correlation with log(1 + Branches)")
    plt.xlabel("Feature Index")
    plt.ylabel("Correlation")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Frequency plot
    # --------------------------------------------------------

    plt.figure(figsize=(10, 4))
    plt.bar(range(n_features), freq)
    plt.title("Feature Observation Frequency")
    plt.xlabel("Feature Index")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def solver_status_breakdown(dataset):
    """
    Print breakdown of solver statuses and feasibility statistics.
    """

    stats = [d["solver_stats"] for d in dataset]

    statuses = [s.status for s in stats]
    solved = np.array([s.solved for s in stats])

    branches = np.array([s.branches for s in stats], dtype=float)
    conflicts = np.array([s.conflicts for s in stats], dtype=float)
    runtime = np.array([s.runtime for s in stats], dtype=float)

    # --------------------------------------------------------
    # Basic counts
    # --------------------------------------------------------

    total = len(dataset)
    solved_count = np.sum(solved)
    unsolved_count = total - solved_count

    print("=" * 70)
    print("SOLVER STATUS BREAKDOWN")
    print("=" * 70)

    print(f"Total instances        : {total}")
    print(f"Solved (feasible)      : {solved_count}")
    print(f"Unsolved              : {unsolved_count}")
    print(f"Solve rate            : {solved_count / total:.4f}")

    print("\n")

    # --------------------------------------------------------
    # Status distribution
    # --------------------------------------------------------

    unique_statuses, counts = np.unique(statuses, return_counts=True)

    print("Status distribution:")
    for s, c in zip(unique_statuses, counts):
        print(f"  {s:<12} : {c}")

    print("\n")

    # --------------------------------------------------------
    # Conditional diagnostics
    # --------------------------------------------------------

    if solved_count > 0:

        print("Solved instances statistics:")
        print(f"  Avg branches     : {branches[solved].mean():.2f}")
        print(f"  Avg conflicts    : {conflicts[solved].mean():.2f}")
        print(f"  Avg runtime      : {runtime[solved].mean():.4f} s")

    if unsolved_count > 0:

        print("\nUnsolved instances statistics:")
        print(f"  Avg branches     : {branches[~solved].mean():.2f}")
        print(f"  Avg conflicts    : {conflicts[~solved].mean():.2f}")
        print(f"  Avg runtime      : {runtime[~solved].mean():.4f} s")

    print("=" * 70)



def main():

    dataset = load_dataset()

    dataset_summary(dataset)

    solver_status_breakdown(dataset)

    plot_solver_histograms(dataset)

    plot_dataset_histograms(dataset)

    plot_correlation_heatmap(dataset)

    plot_solver_relationships(dataset)

    plot_observation_relationships(dataset)

    plot_phase_transition(dataset)

    feature_audit(dataset)


if __name__ == "__main__":
    main()