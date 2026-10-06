"""Plot paired outer-fold Macro-F1 values from the experiment CSVs."""

import csv
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "main_experiment"
DATASETS = ("NEWBORN200", "NICU50")
TASK_LABELS = {
    "audio": "Audio",
    "motion": "Motion",
    "face": "Face",
    "early_fusion": "Early Fusion",
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paired_values(fold_rows, dataset, baseline_task):
    selected = {
        (row["task"], int(row["outer_fold"])): float(row["macro_f1"]) * 100
        for row in fold_rows
        if row["dataset"] == dataset
        and row["model"] == "lgbm"
        and row["task"] in (baseline_task, "early_fusion")
    }
    baseline = [selected[(baseline_task, fold)] for fold in range(10)]
    early = [selected[("early_fusion", fold)] for fold in range(10)]
    return baseline, early


def main():
    fold_rows = read_csv(RESULTS_DIR / "fold.csv")
    paired_rows = read_csv(RESULTS_DIR / "paired.csv")
    paired_by_dataset = {row["dataset"]: row for row in paired_rows}

    values = {}
    all_values = []
    for dataset in DATASETS:
        baseline_task = paired_by_dataset[dataset]["baseline_task"]
        baseline, early = paired_values(fold_rows, dataset, baseline_task)
        values[dataset] = (baseline_task, baseline, early)
        all_values.extend(baseline)
        all_values.extend(early)

    span = max(all_values) - min(all_values)
    padding = max(2.0, span * 0.08)
    y_limits = (max(0.0, min(all_values) - padding), min(100.0, max(all_values) + padding))

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    for axis, dataset in zip(axes, DATASETS):
        baseline_task, baseline, early = values[dataset]
        for baseline_value, early_value in zip(baseline, early):
            axis.plot(
                [0, 1],
                [baseline_value, early_value],
                color="0.75",
                linewidth=0.9,
                zorder=1,
            )
        axis.scatter(
            [0] * 10,
            baseline,
            color="0.35",
            edgecolor="white",
            linewidth=0.3,
            s=18,
            alpha=0.9,
            zorder=3,
        )
        axis.scatter(
            [1] * 10,
            early,
            color="#0072B2",
            edgecolor="white",
            linewidth=0.3,
            s=18,
            alpha=0.9,
            zorder=3,
        )
        axis.set_xticks([0, 1])
        axis.set_xticklabels([TASK_LABELS[baseline_task], "Early Fusion"])
        axis.set_xlim(-0.28, 1.28)
        axis.set_ylim(*y_limits)
        axis.grid(axis="y", color="0.9", linewidth=0.7)
    axes[0].set_ylabel("Macro-F1 (%)")
    figure.tight_layout()
    output_path = RESULTS_DIR / "paired_macro_f1.pdf"
    figure.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    main()
