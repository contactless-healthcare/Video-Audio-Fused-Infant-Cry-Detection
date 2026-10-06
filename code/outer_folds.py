"""Load fixed subject-exclusive outer folds from a CSV manifest."""

import csv
from pathlib import Path

import numpy as np


MANIFEST_NAME = "outer_folds.csv"
MANIFEST_COLUMNS = ["record_id", "subject_id", "outer_fold"]


def subject_id_from_record(record_id):
    return str(record_id).split("_", 1)[0]


def validate_splits(splits, labels, record_ids, n_splits):
    labels = np.asarray(labels, dtype=int)
    subjects = np.asarray(
        [subject_id_from_record(record_id) for record_id in record_ids],
        dtype=object,
    )
    if len(splits) != n_splits:
        raise ValueError(f"Expected {n_splits} outer folds, found {len(splits)}.")

    for fold, (train_indices, test_indices) in enumerate(splits):
        if set(subjects[train_indices]) & set(subjects[test_indices]):
            raise ValueError(f"Outer fold {fold} has overlapping subjects.")
        if len(np.unique(labels[train_indices])) != 2:
            raise ValueError(f"Outer fold {fold} training partition is single-class.")
        if len(np.unique(labels[test_indices])) != 2:
            raise ValueError(f"Outer fold {fold} test partition is single-class.")


def load_outer_splits(manifest_path, record_ids, labels, n_splits):
    """Map the fixed recording assignments onto the current window order."""
    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != MANIFEST_COLUMNS:
            raise ValueError(f"Outer-fold CSV columns must be {MANIFEST_COLUMNS}.")
        rows = list(reader)

    fold_by_record = {}
    fold_by_subject = {}
    for row in rows:
        record_id = row["record_id"]
        subject_id = row["subject_id"]
        fold = int(row["outer_fold"])
        if record_id in fold_by_record:
            raise ValueError(f"Duplicate recording in outer-fold CSV: {record_id}")
        if fold not in range(n_splits):
            raise ValueError(f"Invalid outer fold for recording: {record_id}")
        if subject_id_from_record(record_id) != subject_id:
            raise ValueError(f"Subject does not match recording: {record_id}")
        if subject_id in fold_by_subject and fold_by_subject[subject_id] != fold:
            raise ValueError(f"Subject is assigned to multiple outer folds: {subject_id}")
        fold_by_record[record_id] = fold
        fold_by_subject[subject_id] = fold

    current_records = set(map(str, record_ids))
    if current_records != set(fold_by_record):
        raise ValueError("Current recordings do not match the outer-fold CSV.")

    window_folds = np.asarray(
        [fold_by_record[str(record_id)] for record_id in record_ids], dtype=int
    )
    splits = [
        (np.flatnonzero(window_folds != fold), np.flatnonzero(window_folds == fold))
        for fold in range(n_splits)
    ]
    validate_splits(splits, labels, record_ids, n_splits)
    return splits
