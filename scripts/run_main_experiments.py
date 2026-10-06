"""Run fixed-fold classification experiments and paired statistical comparisons."""

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import PermutationMethod, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import config
import data_loader
import outer_folds
from evaluator import ModelEvaluator
from main import execute_evaluations


DATASETS = ("NEWBORN200", "NICU50")
MODELS = ("svm", "rf", "lgbm")
TASKS = ("audio", "motion", "face", "early_fusion", "late_fusion")
UNIMODAL_TASKS = ("audio", "motion", "face")
METRIC_COLUMNS = {
    "accuracy": "accuracy",
    "macro_precision": "precision",
    "macro_recall": "recall",
    "specificity": "specificity",
    "macro_f1": "f1-score",
}


def run_dataset(dataset):
    """Run all classifiers and tasks for one dataset."""
    config.dataDir = str(PROJECT_ROOT / "dataset" / dataset / "data")
    participant_ids, feature_sets, _, labels, _ = data_loader.load_data(False)
    dataset_root = Path(config.dataDir).parent
    splits = outer_folds.load_outer_splits(
        dataset_root / outer_folds.MANIFEST_NAME,
        participant_ids,
        labels,
        config.n_splits,
    )

    rows = []
    for model in MODELS:
        evaluator = ModelEvaluator(splits)
        task_metrics = execute_evaluations(
            evaluator,
            feature_sets,
            labels,
            participant_ids,
            model,
            tasks=TASKS,
        )
        for task in TASKS:
            metrics = task_metrics[task]
            for outer_fold in range(config.n_splits):
                row = {
                    "dataset": dataset,
                    "model": model,
                    "task": task,
                    "outer_fold": outer_fold,
                }
                row.update(
                    {
                        output_name: float(metrics[metric_name][outer_fold])
                        for output_name, metric_name in METRIC_COLUMNS.items()
                    }
                )
                rows.append(row)
    return rows


def summarize(fold_rows):
    """Compute the ten-fold mean and sample standard deviation."""
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            for task in TASKS:
                selected = [
                    row
                    for row in fold_rows
                    if row["dataset"] == dataset
                    and row["model"] == model
                    and row["task"] == task
                ]
                summary = {
                    "dataset": dataset,
                    "model": model,
                    "task": task,
                }
                for metric in METRIC_COLUMNS:
                    values = np.asarray(
                        [row[metric] for row in selected], dtype=float
                    )
                    summary[f"{metric}_mean"] = float(np.mean(values))
                    summary[f"{metric}_std"] = float(np.std(values, ddof=1))
                rows.append(summary)
    return rows


def select_best_unimodal(summary_rows, dataset):
    """Select the LGBM unimodal task with the highest mean Macro-F1."""
    candidates = [
        row
        for row in summary_rows
        if row["dataset"] == dataset
        and row["model"] == "lgbm"
        and row["task"] in UNIMODAL_TASKS
    ]
    return min(
        candidates,
        key=lambda row: (
            -row["macro_f1_mean"],
            UNIMODAL_TASKS.index(row["task"]),
        ),
    )["task"]


def paired_tests(fold_rows, summary_rows):
    """Compare LGBM Early Fusion with the best unimodal result."""
    by_key = {
        (row["dataset"], row["model"], row["task"], row["outer_fold"]): row
        for row in fold_rows
    }
    rows = []
    for dataset in DATASETS:
        baseline_task = select_best_unimodal(summary_rows, dataset)
        differences = np.asarray(
            [
                by_key[(dataset, "lgbm", "early_fusion", fold)]["macro_f1"]
                - by_key[(dataset, "lgbm", baseline_task, fold)]["macro_f1"]
                for fold in range(config.n_splits)
            ],
            dtype=float,
        )
        differences = np.round(differences, decimals=12)
        result = wilcoxon(
            differences,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method=PermutationMethod(n_resamples=np.inf),
        )
        rows.append(
            {
                "dataset": dataset,
                "model": "lgbm",
                "fusion_task": "early_fusion",
                "baseline_task": baseline_task,
                "metric": "macro_f1",
                "mean_difference_pp": float(np.mean(differences) * 100),
                "wilcoxon_w": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )
    return rows


def write_csv(path, rows):
    """Write rows with stable column order and LF line endings."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    fold_rows = []
    for dataset in DATASETS:
        fold_rows.extend(run_dataset(dataset))

    summary_rows = summarize(fold_rows)
    paired_rows = paired_tests(fold_rows, summary_rows)
    output_dir = PROJECT_ROOT / "results" / "main_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "fold.csv", fold_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "paired.csv", paired_rows)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
