#!/usr/bin/env python3
"""Run the active GUIDE budget-aware SOC queue-triage experiment.

This is the publication pipeline for the GUIDE study.  It consumes the
incident-level Parquet files produced by ``build_guide_incidents.py``; it does
not read the raw evidence CSVs and it does not use the retired AIT-ADS labels.

The program fits organization-disjoint model, calibration-fit, and
calibration-selection partitions; compares a linear baseline with a nonlinear
histogram gradient-boosting model; and selects raw or temperature-scaled
probabilities without consulting the test set. It then evaluates
matched-budget queue policies. Expert QueueRank is used only after decisions
have been generated.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import duckdb
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    from scipy import sparse
    from scipy.optimize import minimize_scalar
    from scipy.special import expit
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_recall_fscore_support,
        roc_auc_score,
    )
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from threadpoolctl import threadpool_limits
except ImportError as exc:  # pragma: no cover - exercised on user machines
    raise SystemExit(
        "Missing dependency. Run: py -m pip install duckdb numpy pandas scipy "
        "scikit-learn matplotlib"
    ) from exc


VERSION = "0.6.0"
STUDY_TITLE = (
    "Budget-Aware AI Decision Intelligence from Cybersecurity Incident Data: "
    "Cost–Safety–Priority Trade-offs in SOC Queue Triage"
)
SEED = 2026
LABELS = ("TruePositive", "BenignPositive", "FalsePositive")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ACTIONS = ("close", "defer", "review_now")
CLOSE, DEFER, REVIEW = 0, 1, 2

# Normalized decision-model units, not monetary claims. Rows are actions and
# columns are TruePositive, BenignPositive, FalsePositive.
COST_SCENARIOS: dict[str, np.ndarray] = {
    "low": np.array([[40.0, 2.0, 0.0], [10.0, 3.0, 6.0], [3.0, 3.0, 3.0]]),
    "medium": np.array(
        [[100.0, 4.0, 0.0], [25.0, 6.0, 12.0], [5.0, 5.0, 5.0]]
    ),
    "high": np.array(
        [[250.0, 10.0, 0.0], [60.0, 15.0, 24.0], [8.0, 8.0, 8.0]]
    ),
}
DEPLOYMENT_COST_WIDTHS = (0.0, 0.25, 0.50)
QUEUE_SIZE_BINS = (-np.inf, 9, 49, 199, np.inf)
QUEUE_SIZE_LABELS = ("1-9", "10-49", "50-199", "200+")

KEY_COLUMNS = ["OrgId", "IncidentId", "Id"]
TARGET = "IncidentGrade"
EVALUATION_ONLY = "Usage"
HIGH_CARDINALITY = ["first_detector_id", "first_alert_title"]
LOW_CARDINALITY = [
    "first_category",
    "first_entity_type",
    "first_evidence_role",
    "first_os_family",
    "first_antispam_direction",
]
CYCLIC_SOURCE = ["first_hour", "first_day_of_week", "first_month"]
BASE_NUMERIC = [
    "evidence_rows",
    "alert_count",
    "detector_count",
    "alert_title_count",
    "category_count",
    "mitre_signature_count",
    "entity_type_count",
    "evidence_role_count",
    "device_count",
    "sha256_count",
    "ip_count",
    "url_count",
    "account_count",
    "network_message_count",
    "email_cluster_count",
    "registry_key_count",
    "application_count",
    "threat_family_count",
    "file_name_count",
    "resource_count",
    "country_count",
    "state_count",
    "city_count",
    "os_family_count",
    "os_version_count",
    "mitre_present_fraction",
    "threat_family_present_fraction",
    "email_cluster_present_fraction",
    "antispam_present_fraction",
    "duration_seconds",
]
PRESENCE_NUMERIC = {
    "mitre_present_fraction",
    "threat_family_present_fraction",
    "email_cluster_present_fraction",
    "antispam_present_fraction",
}


def parse_float_list(value: str) -> list[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("Expected a comma-separated numeric list")
    return result


def parse_int_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("Expected a comma-separated integer list")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run calibrated GUIDE classification and SOC queue policies."
    )
    parser.add_argument("--train-parquet", required=True, type=Path)
    parser.add_argument("--test-parquet", required=True, type=Path)
    parser.add_argument("--rankings", required=True, type=Path)
    parser.add_argument("--build-report", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("guide_results"))
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--hgb-iterations", type=int, default=250)
    parser.add_argument("--tv-radius", type=float, default=0.025)
    parser.add_argument(
        "--automation-cost-width",
        "--cost-width",
        dest="automation_cost_width",
        type=float,
        default=0.25,
        help=(
            "One-sided relative uncertainty applied to close/defer costs only; "
            "review cost remains fixed. --cost-width is retained as a legacy alias."
        ),
    )
    parser.add_argument(
        "--calibration-selection-fraction",
        type=float,
        default=0.50,
        help="Organization-disjoint share of the calibration holdout used only to select raw versus temperature probabilities.",
    )
    parser.add_argument("--close-threshold", type=float, default=0.10)
    parser.add_argument(
        "--closure-safety",
        choices=("tp", "tp_bp"),
        default="tp",
        help="Primary closure harm definition; tp_bp is the stricter sensitivity variant.",
    )
    parser.add_argument("--priority-lambdas", type=parse_float_list, default=[5.0, 10.0, 20.0])
    parser.add_argument("--full-budget-fractions", type=parse_float_list, default=[0.01, 0.05, 0.10, 0.20])
    parser.add_argument("--ranked-budgets", type=parse_int_list, default=[1, 3, 5, 10])
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--primary-lambda", type=float, default=10.0)
    parser.add_argument("--max-train-incidents", type=int, default=0)
    parser.add_argument("--max-test-incidents", type=int, default=0)
    parser.add_argument("--skip-linear", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-partial-rankings", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def verify_inputs(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.train_parquet, args.test_parquet, args.rankings, args.build_report):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(args.build_report.read_text(encoding="utf-8"))
    if report.get("build_version") != "0.2.0":
        raise ValueError(f"Unsupported build report version: {report.get('build_version')!r}")
    checks = [
        (args.train_parquet, report["train"]["file"]["size_bytes"], "train Parquet"),
        (args.test_parquet, report["test"]["file"]["size_bytes"], "test Parquet"),
    ]
    for path, expected, label in checks:
        if path.stat().st_size != int(expected):
            raise ValueError(
                f"{label} differs from the validated build: {path.stat().st_size} != {expected}"
            )
    if int(report["train"]["duplicate_keys"]) != 0 or int(report["test"]["duplicate_keys"]) != 0:
        raise ValueError("Build report contains duplicate incident keys")
    if not args.allow_partial_rankings and int(report["ranking_join_coverage"]["unmatched"]) != 0:
        raise ValueError("Build report did not match every ranked incident")
    return report


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}; use --overwrite")
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def load_incidents(path: Path, is_test: bool) -> pd.DataFrame:
    requested = (
        KEY_COLUMNS
        + [TARGET, "first_timestamp", "last_timestamp"]
        + ([EVALUATION_ONLY] if is_test else [])
        + BASE_NUMERIC
        + HIGH_CARDINALITY
        + LOW_CARDINALITY
        + CYCLIC_SOURCE
    )
    connection = duckdb.connect(database=":memory:")
    available = {
        row[0]
        for row in connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{str(path.resolve()).replace("'", "''")}')"
        ).fetchall()
    }
    missing = sorted(set(requested) - available)
    if missing:
        raise ValueError(f"Incident Parquet is missing required columns: {missing}")
    selected = ", ".join(f'"{column}"' for column in requested)
    frame = connection.execute(
        f"SELECT {selected} FROM read_parquet('{str(path.resolve()).replace("'", "''")}')"
    ).df()
    connection.close()
    for column in KEY_COLUMNS:
        frame[column] = frame[column].astype("string")
    frame[TARGET] = frame[TARGET].astype("string")
    if is_test:
        frame[EVALUATION_ONLY] = frame[EVALUATION_ONLY].astype("string")
    frame["first_timestamp"] = pd.to_datetime(frame["first_timestamp"], utc=True)
    frame["last_timestamp"] = pd.to_datetime(frame["last_timestamp"], utc=True)
    return frame


def stratified_limit(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if maximum <= 0 or len(frame) <= maximum:
        return frame.reset_index(drop=True)
    fractions = frame[TARGET].value_counts(normalize=True)
    pieces = []
    allocated = 0
    labels = list(fractions.index)
    for index, label in enumerate(labels):
        group = frame[frame[TARGET] == label]
        if index == len(labels) - 1:
            take = maximum - allocated
        else:
            take = min(len(group), max(1, int(round(maximum * fractions[label]))))
        pieces.append(group.sample(n=take, random_state=seed + index))
        allocated += take
    return pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


class IncidentFeatureBuilder:
    """Fit-only transformations for incident summaries.

    Raw detector and alert-title codes are converted to training-frequency and
    rarity statistics. Raw entity identifiers never enter the model.
    """

    def __init__(self, max_categories: int = 200) -> None:
        self.max_categories = max_categories
        self.caps: dict[str, float] = {}
        self.medians: dict[str, float] = {}
        self.high_counts: dict[str, dict[str, int]] = {}
        self.low_maps: dict[str, dict[str, int]] = {}
        self.n_fit = 0
        self.feature_names: list[str] = []
        self.categorical_mask: np.ndarray | None = None

    @staticmethod
    def _clean_category(series: pd.Series) -> pd.Series:
        return series.astype("string").fillna("__MISSING__")

    def fit(self, frame: pd.DataFrame) -> "IncidentFeatureBuilder":
        self.n_fit = len(frame)
        for column in BASE_NUMERIC:
            values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            median = float(values.median()) if values.notna().any() else 0.0
            cap = float(values.quantile(0.999)) if values.notna().any() else median
            self.medians[column] = median
            self.caps[column] = max(median, cap)
        for column in HIGH_CARDINALITY:
            counts = self._clean_category(frame[column]).value_counts(dropna=False)
            self.high_counts[column] = {str(key): int(value) for key, value in counts.items()}
        for column in LOW_CARDINALITY:
            counts = self._clean_category(frame[column]).value_counts(dropna=False)
            kept = list(counts.head(self.max_categories).index)
            self.low_maps[column] = {str(value): index for index, value in enumerate(kept)}
        return self

    def _continuous(self, frame: pd.DataFrame) -> tuple[list[np.ndarray], list[str]]:
        arrays: list[np.ndarray] = []
        names: list[str] = []
        for column in BASE_NUMERIC:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            values = np.nan_to_num(values, nan=self.medians[column], posinf=self.caps[column], neginf=0.0)
            values = np.clip(values, 0.0, self.caps[column])
            if column not in PRESENCE_NUMERIC:
                values = np.log1p(values)
                name = f"log1p_{column}"
            else:
                name = column
            arrays.append(values.astype(np.float32))
            names.append(name)

        periods = {"first_hour": 24.0, "first_day_of_week": 7.0, "first_month": 12.0}
        offsets = {"first_hour": 0.0, "first_day_of_week": 0.0, "first_month": 1.0}
        for column in CYCLIC_SOURCE:
            values = pd.to_numeric(frame[column], errors="coerce").fillna(offsets[column]).to_numpy(dtype=float)
            angle = 2.0 * np.pi * (values - offsets[column]) / periods[column]
            arrays.extend([np.sin(angle).astype(np.float32), np.cos(angle).astype(np.float32)])
            names.extend([f"{column}_sin", f"{column}_cos"])

        denominator = max(1, self.n_fit)
        for column in HIGH_CARDINALITY:
            cleaned = self._clean_category(frame[column])
            counts = cleaned.map(self.high_counts[column]).fillna(0).to_numpy(dtype=float)
            arrays.append(np.log1p(counts).astype(np.float32))
            arrays.append((-np.log((counts + 1.0) / (denominator + 1.0))).astype(np.float32))
            names.extend([f"{column}_frequency_log1p", f"{column}_rarity"])
        return arrays, names

    def transform_hgb(self, frame: pd.DataFrame) -> np.ndarray:
        arrays, names = self._continuous(frame)
        categorical_mask = [False] * len(arrays)
        for column in LOW_CARDINALITY:
            cleaned = self._clean_category(frame[column])
            mapped = cleaned.map(self.low_maps[column]).to_numpy(dtype=float)
            arrays.append(mapped.astype(np.float32))
            names.append(column)
            categorical_mask.append(True)
        self.feature_names = names
        self.categorical_mask = np.asarray(categorical_mask, dtype=bool)
        return np.column_stack(arrays).astype(np.float32, copy=False)

    def priority_score(self, frame: pd.DataFrame) -> pd.Series:
        detector_counts = self._clean_category(frame[HIGH_CARDINALITY[0]]).map(
            self.high_counts[HIGH_CARDINALITY[0]]
        ).fillna(0)
        title_counts = self._clean_category(frame[HIGH_CARDINALITY[1]]).map(
            self.high_counts[HIGH_CARDINALITY[1]]
        ).fillna(0)
        rarity_detector = -np.log((detector_counts + 1.0) / (self.n_fit + 1.0))
        rarity_title = -np.log((title_counts + 1.0) / (self.n_fit + 1.0))
        raw = (
            0.30 * np.log1p(pd.to_numeric(frame["evidence_rows"], errors="coerce").fillna(0))
            + 0.20 * np.log1p(pd.to_numeric(frame["alert_count"], errors="coerce").fillna(0))
            + 0.25 * rarity_detector
            + 0.25 * rarity_title
        )
        temp = pd.DataFrame({"OrgId": frame["OrgId"].to_numpy(), "raw": raw.to_numpy()})
        return temp.groupby("OrgId", sort=False)["raw"].rank(method="average", pct=True)

    def linear_matrices(
        self, fit: pd.DataFrame, calibration: pd.DataFrame, test: pd.DataFrame
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, dict[str, Any]]:
        fit_hgb = self.transform_hgb(fit)
        cal_hgb = self.transform_hgb(calibration)
        test_hgb = self.transform_hgb(test)
        assert self.categorical_mask is not None
        numeric_mask = ~self.categorical_mask
        scaler = StandardScaler(with_mean=False)
        fit_numeric = scaler.fit_transform(fit_hgb[:, numeric_mask])
        cal_numeric = scaler.transform(cal_hgb[:, numeric_mask])
        test_numeric = scaler.transform(test_hgb[:, numeric_mask])

        def normalized_categories(frame: pd.DataFrame) -> pd.DataFrame:
            result = {}
            for column in LOW_CARDINALITY:
                clean = self._clean_category(frame[column])
                keep = set(self.low_maps[column])
                result[column] = clean.where(clean.isin(keep), "__OTHER__")
            return pd.DataFrame(result)

        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
        fit_cat = encoder.fit_transform(normalized_categories(fit))
        cal_cat = encoder.transform(normalized_categories(calibration))
        test_cat = encoder.transform(normalized_categories(test))
        metadata = {
            "continuous_features": [
                name for name, is_cat in zip(self.feature_names, self.categorical_mask) if not is_cat
            ],
            "one_hot_features": int(fit_cat.shape[1]),
        }
        return (
            sparse.hstack([fit_numeric, fit_cat], format="csr", dtype=np.float32),
            sparse.hstack([cal_numeric, cal_cat], format="csr", dtype=np.float32),
            sparse.hstack([test_numeric, test_cat], format="csr", dtype=np.float32),
            metadata,
        )


class TemperatureScaler:
    def __init__(self) -> None:
        self.temperature = 1.0
        self.calibration_loss_before: float | None = None
        self.calibration_loss_after: float | None = None

    @staticmethod
    def _transform(probabilities: np.ndarray, temperature: float) -> np.ndarray:
        logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
        logits -= logits.max(axis=1, keepdims=True)
        values = np.exp(logits)
        return values / values.sum(axis=1, keepdims=True)

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        self.calibration_loss_before = float(log_loss(labels, probabilities, labels=np.arange(3)))

        def objective(log_temperature: float) -> float:
            temperature = float(np.exp(log_temperature))
            calibrated = self._transform(probabilities, temperature)
            return float(log_loss(labels, calibrated, labels=np.arange(3)))

        result = minimize_scalar(objective, bounds=(math.log(0.2), math.log(100.0)), method="bounded")
        candidate = float(np.exp(result.x)) if result.success else 1.0
        candidate_loss = objective(math.log(candidate))
        if candidate_loss <= self.calibration_loss_before + 1e-10:
            self.temperature = candidate
            self.calibration_loss_after = candidate_loss
        else:
            self.temperature = 1.0
            self.calibration_loss_after = self.calibration_loss_before
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        return self._transform(probabilities, self.temperature)


@dataclass
class ProbabilitySelection:
    scaler: TemperatureScaler
    temperature_calibration_fit: np.ndarray
    temperature_selection: np.ndarray
    temperature_test: np.ndarray
    selected_test: np.ndarray
    selected_variant: str
    selection_log_loss_raw: float
    selection_log_loss_temperature: float


def select_probability_variant(
    raw_calibration_fit: np.ndarray,
    labels_calibration_fit: np.ndarray,
    raw_selection: np.ndarray,
    labels_selection: np.ndarray,
    raw_test: np.ndarray,
) -> ProbabilitySelection:
    """Fit temperature on one org partition and select it on another.

    The held-out selection partition prevents reporting the temperature
    optimizer's training objective as evidence that calibration generalizes.
    The test labels never participate in this choice.
    """

    scaler = TemperatureScaler().fit(raw_calibration_fit, labels_calibration_fit)
    temperature_calibration_fit = scaler.transform(raw_calibration_fit)
    temperature_selection = scaler.transform(raw_selection)
    temperature_test = scaler.transform(raw_test)
    raw_loss = float(
        log_loss(labels_selection, raw_selection, labels=np.arange(len(LABELS)))
    )
    temperature_loss = float(
        log_loss(
            labels_selection,
            temperature_selection,
            labels=np.arange(len(LABELS)),
        )
    )
    selected_variant = "temperature" if temperature_loss < raw_loss else "raw"
    selected_test = temperature_test if selected_variant == "temperature" else raw_test
    return ProbabilitySelection(
        scaler=scaler,
        temperature_calibration_fit=temperature_calibration_fit,
        temperature_selection=temperature_selection,
        temperature_test=temperature_test,
        selected_test=selected_test,
        selected_variant=selected_variant,
        selection_log_loss_raw=raw_loss,
        selection_log_loss_temperature=temperature_loss,
    )


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            result += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return result


def classification_metrics(
    probabilities: np.ndarray, labels: np.ndarray, model: str, split: str
) -> dict[str, Any]:
    prediction = probabilities.argmax(axis=1)
    one_hot = np.eye(3, dtype=float)[labels]
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, prediction, labels=np.arange(3), zero_division=0
    )
    row: dict[str, Any] = {
        "model": model,
        "split": split,
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "log_loss": float(log_loss(labels, probabilities, labels=np.arange(3))),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "top_label_ece": float(expected_calibration_error(probabilities, labels)),
        "macro_pr_auc": float(average_precision_score(one_hot, probabilities, average="macro")),
        "macro_roc_auc_ovr": float(roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr")),
    }
    for index, label in enumerate(LABELS):
        stem = label.lower()
        row[f"precision_{stem}"] = float(precision[index])
        row[f"recall_{stem}"] = float(recall[index])
        row[f"f1_{stem}"] = float(f1[index])
        row[f"support_{stem}"] = int(support[index])
    row["confusion_matrix"] = json.dumps(confusion_matrix(labels, prediction, labels=np.arange(3)).tolist())
    return row


def stable_ovr_probabilities(model: SGDClassifier, matrix: sparse.csr_matrix) -> np.ndarray:
    """Convert one-vs-rest SGD scores to finite normalized probabilities."""
    scores = np.asarray(model.decision_function(matrix), dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(LABELS):
        raise RuntimeError(f"Unexpected linear decision shape: {scores.shape}")
    values = expit(scores)
    totals = values.sum(axis=1, keepdims=True)
    bad = totals[:, 0] <= 1e-12
    if bad.any():
        fallback = scores[bad] - scores[bad].max(axis=1, keepdims=True)
        values[bad] = np.exp(np.clip(fallback, -700.0, 0.0))
        totals = values.sum(axis=1, keepdims=True)
    probabilities = values / totals
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Linear baseline produced non-finite probabilities")
    return probabilities


def worst_case_tv_expectation(
    probabilities: np.ndarray, costs: np.ndarray, radius: float
) -> np.ndarray:
    if not 0.0 <= radius <= 1.0:
        raise ValueError("TV radius must be in [0, 1]")
    shifted = np.asarray(probabilities, dtype=float).copy()
    low_order = np.argsort(costs)
    high_order = low_order[::-1]
    remaining = np.full(len(shifted), radius, dtype=float)
    for low in low_order:
        for high in high_order:
            if costs[high] <= costs[low]:
                continue
            amount = np.minimum.reduce(
                [remaining, shifted[:, low], 1.0 - shifted[:, high]]
            )
            shifted[:, low] -= amount
            shifted[:, high] += amount
            remaining -= amount
    return shifted @ costs


def action_risks(
    probabilities: np.ndarray,
    costs: np.ndarray,
    robust: bool,
    tv_radius: float,
    automation_cost_width: float,
) -> np.ndarray:
    decision_costs = np.asarray(costs, dtype=float).copy()
    if robust:
        # Cost uncertainty is one-sided and action-specific: automation harm
        # (close/defer) may be underestimated, while the known analyst-review
        # cost stays fixed. Uniformly scaling every action would cancel from
        # action comparisons and would not represent cost uncertainty.
        decision_costs[[CLOSE, DEFER], :] *= 1.0 + automation_cost_width
    if not robust:
        return probabilities @ decision_costs.T
    return np.column_stack(
        [
            worst_case_tv_expectation(probabilities, decision_costs[action], tv_radius)
            for action in range(3)
        ]
    )


def deployment_cost_matrix(costs: np.ndarray, automation_cost_width: float) -> np.ndarray:
    """Return an outcome-evaluation matrix for a declared deployment condition."""
    if automation_cost_width < 0:
        raise ValueError("Deployment automation-cost width must be nonnegative")
    deployed = np.asarray(costs, dtype=float).copy()
    deployed[[CLOSE, DEFER], :] *= 1.0 + automation_cost_width
    return deployed


def oracle_cost_for_queue(
    labels: np.ndarray, costs: np.ndarray, review_budget: int
) -> tuple[float, int]:
    """Minimum realized cost with known outcomes and an at-most review budget.

    The oracle is deliberately unattainable. It is a regret reference, not a
    deployable baseline. It reviews only incidents for which review improves
    realized cost and therefore may use fewer than ``review_budget`` slots.
    """
    y = np.asarray(labels, dtype=np.int8)
    budget = int(np.clip(review_budget, 0, len(y)))
    automatic = np.minimum(costs[CLOSE, y], costs[DEFER, y])
    review = costs[REVIEW, y]
    gains = automatic - review
    chosen = np.flatnonzero(gains > 0)
    if budget and len(chosen) > budget:
        chosen = chosen[np.argpartition(gains[chosen], len(chosen) - budget)[-budget:]]
    elif budget == 0:
        chosen = np.empty(0, dtype=int)
    realized = automatic.copy()
    realized[chosen] = review[chosen]
    return float(realized.sum()), int(len(chosen))


def peak_rss_bytes() -> int | None:
    """Best-effort process peak resident memory on Windows, Linux, and macOS."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            )
            return int(counters.PeakWorkingSetSize) if ok else None

        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if platform.system() == "Darwin" else value * 1024
    except Exception:
        return None


def entropy(probabilities: np.ndarray) -> np.ndarray:
    return -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(axis=1)


@dataclass
class PolicyOutput:
    actions: np.ndarray
    review_scores: np.ndarray
    predicted_risk: np.ndarray


def apply_policy_to_queue(
    probabilities: np.ndarray,
    priority: np.ndarray,
    alert_count: np.ndarray,
    timestamps: np.ndarray,
    budget: int,
    policy: str,
    costs: np.ndarray,
    tv_radius: float,
    automation_cost_width: float,
    close_threshold: float,
    closure_safety: str,
    priority_lambda: float,
    seed: int,
) -> PolicyOutput:
    n = len(probabilities)
    budget = int(np.clip(budget, 0, n))
    if policy == "priority_defer_only":
        # True no-probability ablation: incident-level probabilities do not
        # influence review selection or the automatic action. Unreviewed
        # incidents are conservatively deferred rather than auto-closed.
        score = np.asarray(priority, dtype=float)
        actions = np.full(n, DEFER, dtype=np.int8)
        if budget:
            chosen = np.argpartition(score, n - budget)[n - budget :]
            actions[chosen] = REVIEW
        return PolicyOutput(
            actions=actions,
            review_scores=score,
            predicted_risk=np.full(n, np.nan, dtype=float),
        )

    robust = policy in {"robust_no_priority", "joint_robust", "global_tp_threshold"}
    risks = action_risks(
        probabilities, costs, robust, tv_radius, automation_cost_width
    )
    if robust:
        harmful_probability = probabilities[:, 0]
        if closure_safety == "tp_bp":
            harmful_probability = harmful_probability + probabilities[:, 1]
        elif closure_safety != "tp":
            raise ValueError(f"Unknown closure safety definition: {closure_safety}")
        harmful_upper = np.minimum(1.0, harmful_probability + tv_radius)
        risks[harmful_upper > close_threshold, CLOSE] = np.inf
    base = np.where(risks[:, CLOSE] <= risks[:, DEFER], CLOSE, DEFER)
    base_risk = risks[np.arange(n), base]
    gain = base_risk - risks[:, REVIEW]

    if policy == "random":
        score = np.random.default_rng(seed).random(n)
    elif policy == "arrival_oldest":
        score = -timestamps.astype("datetime64[ns]").astype(np.int64).astype(float)
    elif policy == "alert_count":
        score = alert_count.astype(float)
    elif policy == "tp_probability":
        score = probabilities[:, 0]
    elif policy == "uncertainty":
        score = entropy(probabilities)
    elif policy == "priority_only":
        score = priority
    elif policy in {"nominal_optimizer", "robust_no_priority"}:
        score = gain
    elif policy == "joint_robust":
        score = gain + priority_lambda * priority
    elif policy == "global_tp_threshold":
        score = probabilities[:, 0]
    else:
        raise ValueError(f"Unknown policy: {policy}")

    actions = base.astype(np.int8)
    if policy == "global_tp_threshold":
        actions[score >= priority_lambda] = REVIEW
    elif budget:
        chosen = np.argpartition(score, n - budget)[n - budget :]
        actions[chosen] = REVIEW
    selected_risk = risks[np.arange(n), actions]
    return PolicyOutput(actions=actions, review_scores=score, predicted_risk=selected_risk)


def budget_for_queue(n: int, budget_value: float, exact: bool) -> int:
    if exact:
        return min(n, int(budget_value))
    if budget_value <= 0 or n <= 0:
        return 0
    # A fractional capacity is an upper bound. Rounding up every small queue
    # silently violates the system-wide workload target.
    return min(n, int(np.floor(n * budget_value)))


def allocate_queue_budgets(
    queue_groups: list[tuple[Any, np.ndarray]],
    budget_value: float,
    exact: bool,
    allocation_mode: str = "per_queue_floor",
) -> dict[str, int]:
    """Allocate deterministic queue budgets without exceeding system capacity.

    ``per_queue_floor`` is the preregistered primary rule. The
    ``largest_remainder`` sensitivity preserves a pooled target of
    floor(total incidents * fraction), then assigns residual slots by the
    largest fractional remainder with a deterministic organization tie-break.
    """
    sizes = {str(organization): int(len(index)) for organization, index in queue_groups}
    if exact:
        return {
            organization: budget_for_queue(size, budget_value, True)
            for organization, size in sizes.items()
        }
    if allocation_mode == "per_queue_floor":
        return {
            organization: budget_for_queue(size, budget_value, False)
            for organization, size in sizes.items()
        }
    if allocation_mode != "largest_remainder":
        raise ValueError(f"Unknown budget allocation mode: {allocation_mode}")
    if not 0.0 <= budget_value <= 1.0:
        raise ValueError("A fractional review budget must be in [0, 1]")

    budgets = {
        organization: min(size, int(np.floor(size * budget_value)))
        for organization, size in sizes.items()
    }
    target = int(np.floor(sum(sizes.values()) * budget_value))
    remaining = target - sum(budgets.values())
    order = sorted(
        sizes,
        key=lambda organization: (
            -(sizes[organization] * budget_value - budgets[organization]),
            organization,
        ),
    )
    for organization in order:
        if remaining <= 0:
            break
        if budgets[organization] < sizes[organization]:
            budgets[organization] += 1
            remaining -= 1
    if remaining != 0 or sum(budgets.values()) != target:
        raise RuntimeError("Largest-remainder allocation did not meet pooled capacity")
    return budgets


def expert_ranking_metrics(
    ranks: np.ndarray,
    reviewed: np.ndarray,
    review_scores: np.ndarray,
    budget: int,
) -> dict[str, float]:
    """Return transparent expert-ranking metrics for a reviewed queue."""
    if budget <= 0:
        return {
            "expert_top_budget_coverage": np.nan,
            "expert_precision_at_budget": np.nan,
            "expert_priority_gain_ratio": np.nan,
            "expert_ndcg_at_budget": np.nan,
        }
    selected = np.flatnonzero(reviewed)
    expert_top = ranks <= budget
    intersection = int(np.sum(reviewed & expert_top))
    ideal_gain = float(np.sum(21 - np.arange(1, budget + 1)))
    selected_gain = float(np.sum((21 - ranks)[reviewed]))

    # Policies produce review scores, so the selected set is ordered by score
    # for NDCG. Original row position is a deterministic tie-break and expert
    # rank is never used to construct the ordering.
    ordered = selected[
        np.lexsort((selected, -np.asarray(review_scores, dtype=float)[selected]))
    ]
    discounts = np.log2(np.arange(2, len(ordered) + 2, dtype=float))
    dcg = float(np.sum((21 - ranks[ordered]) / discounts))
    ideal_discounts = np.log2(np.arange(2, budget + 2, dtype=float))
    ideal_dcg = float(
        np.sum((21 - np.arange(1, budget + 1, dtype=float)) / ideal_discounts)
    )
    return {
        "expert_top_budget_coverage": intersection / budget,
        "expert_precision_at_budget": intersection / len(selected)
        if len(selected)
        else np.nan,
        "expert_priority_gain_ratio": selected_gain / ideal_gain
        if ideal_gain
        else np.nan,
        "expert_ndcg_at_budget": dcg / ideal_dcg if ideal_dcg else np.nan,
    }


def fit_global_tp_thresholds(
    probabilities: np.ndarray, target_review_rates: Iterable[float]
) -> dict[float, float]:
    """Freeze capacity-agnostic TP thresholds on validation organizations."""
    tp_probability = np.asarray(probabilities, dtype=float)[:, 0]
    result: dict[float, float] = {}
    for value in target_review_rates:
        rate = float(value)
        if not 0.0 < rate <= 1.0:
            raise ValueError("Target review rates must be in (0, 1]")
        threshold = (
            float("-inf")
            if rate == 1.0
            else float(np.quantile(tp_probability, 1.0 - rate, method="higher"))
        )
        result[round(rate, 10)] = threshold
    return result


def evaluate_policy_grid(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    priority: np.ndarray,
    scope: str,
    budget_values: Iterable[float],
    exact_budgets: bool,
    priority_lambdas: list[float],
    cost_scenario: str,
    tv_radius: float,
    automation_cost_width: float,
    close_threshold: float,
    closure_safety: str,
    seed: int,
    probability_variant: str,
    policy_specs_override: list[tuple[str, str, float]] | None = None,
    evaluate_deployment_stress: bool = False,
    deployment_stress_policies: set[str] | None = None,
    budget_allocation_mode: str = "per_queue_floor",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    costs = COST_SCENARIOS[cost_scenario]
    labels = frame[TARGET].map(LABEL_TO_ID).to_numpy(dtype=np.int8)
    alert_counts = pd.to_numeric(frame["alert_count"], errors="coerce").fillna(0).to_numpy(dtype=float)
    timestamps = frame["first_timestamp"].to_numpy(dtype="datetime64[ns]")
    queue_groups = list(frame.groupby("OrgId", sort=True).indices.items())
    if policy_specs_override is None:
        policy_specs: list[tuple[str, str, float]] = [
            ("random", "random", 0.0),
            ("arrival_oldest", "arrival_oldest", 0.0),
            ("alert_count", "alert_count", 0.0),
            ("tp_probability", "tp_probability", 0.0),
            ("uncertainty", "uncertainty", 0.0),
            ("priority_only", "priority_only", 0.0),
            ("priority_defer_only", "priority_defer_only", 0.0),
            ("nominal_optimizer", "nominal_optimizer", 0.0),
            ("robust_no_priority", "robust_no_priority", 0.0),
        ]
        for value in priority_lambdas:
            policy_specs.append((f"joint_robust_lambda_{value:g}", "joint_robust", value))
    else:
        policy_specs = policy_specs_override

    per_queue_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    stress_rows: list[dict[str, Any]] = []
    has_expert_rank = "QueueRank" in frame.columns

    for budget_value in budget_values:
        queue_budgets = allocate_queue_budgets(
            queue_groups,
            float(budget_value),
            exact_budgets,
            allocation_mode=budget_allocation_mode,
        )
        for policy_number, (label, policy, lambda_value) in enumerate(policy_specs):
            all_actions = np.empty(len(frame), dtype=np.int8)
            all_scores = np.empty(len(frame), dtype=float)
            all_predicted_risk = np.empty(len(frame), dtype=float)
            queue_rows: list[dict[str, Any]] = []
            for queue_number, (organization, index) in enumerate(queue_groups):
                idx = np.asarray(index, dtype=int)
                budget = queue_budgets[str(organization)]
                output = apply_policy_to_queue(
                    probabilities[idx],
                    priority[idx],
                    alert_counts[idx],
                    timestamps[idx],
                    budget,
                    policy,
                    costs,
                    tv_radius,
                    automation_cost_width,
                    close_threshold,
                    closure_safety,
                    lambda_value,
                    seed + 100_000 * policy_number + 101 * queue_number + budget,
                )
                all_actions[idx] = output.actions
                all_scores[idx] = output.review_scores
                all_predicted_risk[idx] = output.predicted_risk

                y = labels[idx]
                realized = costs[output.actions, y]
                oracle_budget = (
                    int(np.sum(output.actions == REVIEW))
                    if policy == "global_tp_threshold"
                    else budget
                )
                oracle_cost_sum, oracle_reviews = oracle_cost_for_queue(
                    y, costs, oracle_budget
                )
                tp = y == LABEL_TO_ID["TruePositive"]
                non_fp = y != LABEL_TO_ID["FalsePositive"]
                row: dict[str, Any] = {
                    "scope": scope,
                    "probability_variant": probability_variant,
                    "organization": str(organization),
                    "policy": label,
                    "cost_scenario": cost_scenario,
                    "budget_value": float(budget_value),
                    "budget_is_exact": bool(exact_budgets),
                    "budget_allocation_mode": (
                        "exact_per_queue" if exact_budgets else budget_allocation_mode
                    ),
                    "queue_size": int(len(idx)),
                    "review_budget": int(budget),
                    "reviews": int(np.sum(output.actions == REVIEW)),
                    "capacity_violation": int(np.sum(output.actions == REVIEW) > budget),
                    "excess_reviews": int(
                        max(0, int(np.sum(output.actions == REVIEW)) - budget)
                    ),
                    "realized_cost_mean": float(realized.mean()),
                    "predicted_robust_risk_mean": (
                        float(np.mean(output.predicted_risk))
                        if np.isfinite(output.predicted_risk).all()
                        else np.nan
                    ),
                    "close_rate": float(np.mean(output.actions == CLOSE)),
                    "defer_rate": float(np.mean(output.actions == DEFER)),
                    "review_rate": float(np.mean(output.actions == REVIEW)),
                    "tp_close_count": int(np.sum((output.actions == CLOSE) & tp)),
                    "tp_count": int(np.sum(tp)),
                    "non_fp_close_count": int(np.sum((output.actions == CLOSE) & non_fp)),
                    "non_fp_count": int(np.sum(non_fp)),
                    "realized_cost_sum": float(realized.sum()),
                    "oracle_cost_sum": oracle_cost_sum,
                    "oracle_cost_mean": oracle_cost_sum / len(idx),
                    "oracle_reviews": oracle_reviews,
                    "regret_sum": float(realized.sum() - oracle_cost_sum),
                    "regret_mean": float((realized.sum() - oracle_cost_sum) / len(idx)),
                }
                if has_expert_rank:
                    ranks = pd.to_numeric(frame.iloc[idx]["QueueRank"], errors="raise").to_numpy(dtype=int)
                    reviewed = output.actions == REVIEW
                    row.update(
                        expert_ranking_metrics(
                            ranks, reviewed, output.review_scores, budget
                        )
                    )
                queue_rows.append(row)

                if evaluate_deployment_stress and (
                    deployment_stress_policies is None
                    or label in deployment_stress_policies
                ):
                    for deployment_scenario, base_deployment_costs in COST_SCENARIOS.items():
                        for deployment_width in DEPLOYMENT_COST_WIDTHS:
                            deployed_costs = deployment_cost_matrix(
                                base_deployment_costs, deployment_width
                            )
                            deployed_realized = deployed_costs[output.actions, y]
                            deployed_oracle, deployed_oracle_reviews = oracle_cost_for_queue(
                                y, deployed_costs, oracle_budget
                            )
                            scale = float(np.max(deployed_costs))
                            deployed_sum = float(deployed_realized.sum())
                            deployed_regret = deployed_sum - deployed_oracle
                            stress_rows.append(
                                {
                                    "scope": scope,
                                    "probability_variant": probability_variant,
                                    "organization": str(organization),
                                    "policy": label,
                                    "decision_cost_scenario": cost_scenario,
                                    "budget_value": float(budget_value),
                                    "budget_is_exact": bool(exact_budgets),
                                    "budget_allocation_mode": (
                                        "exact_per_queue"
                                        if exact_budgets
                                        else budget_allocation_mode
                                    ),
                                    "queue_size": int(len(idx)),
                                    "review_budget": int(budget),
                                    "reviews": int(np.sum(output.actions == REVIEW)),
                                    "deployment_cost_scenario": deployment_scenario,
                                    "deployment_automation_cost_width": float(
                                        deployment_width
                                    ),
                                    "cost_scale": scale,
                                    "realized_cost_sum": deployed_sum,
                                    "realized_cost_mean": deployed_sum / len(idx),
                                    "normalized_realized_cost_sum": deployed_sum / scale,
                                    "normalized_realized_cost_mean": deployed_sum
                                    / (scale * len(idx)),
                                    "oracle_cost_sum": deployed_oracle,
                                    "oracle_cost_mean": deployed_oracle / len(idx),
                                    "oracle_reviews": deployed_oracle_reviews,
                                    "regret_sum": deployed_regret,
                                    "regret_mean": deployed_regret / len(idx),
                                    "normalized_regret_sum": deployed_regret / scale,
                                    "normalized_regret_mean": deployed_regret
                                    / (scale * len(idx)),
                                }
                            )

            queue_frame = pd.DataFrame(queue_rows)
            per_queue_rows.extend(queue_rows)
            tp_total = int(queue_frame["tp_count"].sum())
            non_fp_total = int(queue_frame["non_fp_count"].sum())
            total_n = int(queue_frame["queue_size"].sum())
            aggregate: dict[str, Any] = {
                "scope": scope,
                "probability_variant": probability_variant,
                "policy": label,
                "cost_scenario": cost_scenario,
                "budget_value": float(budget_value),
                "budget_is_exact": bool(exact_budgets),
                "budget_allocation_mode": (
                    "exact_per_queue" if exact_budgets else budget_allocation_mode
                ),
                "queues": int(len(queue_frame)),
                "incidents": total_n,
                "reviews": int(queue_frame["reviews"].sum()),
                "system_review_target": int(queue_frame["review_budget"].sum()),
                "queues_with_zero_review_budget": int(
                    queue_frame["review_budget"].eq(0).sum()
                ),
                "zero_review_budget_rate": float(
                    queue_frame["review_budget"].eq(0).mean()
                ),
                "capacity_violation_queues": int(
                    queue_frame["capacity_violation"].sum()
                ),
                "capacity_violation_rate": float(
                    queue_frame["capacity_violation"].mean()
                ),
                "excess_reviews": int(queue_frame["excess_reviews"].sum()),
                "actual_review_rate": float(queue_frame["reviews"].sum() / total_n),
                "incident_weighted_cost": float(queue_frame["realized_cost_sum"].sum() / total_n),
                "mean_queue_cost": float(queue_frame["realized_cost_mean"].mean()),
                "incident_weighted_oracle_cost": float(
                    queue_frame["oracle_cost_sum"].sum() / total_n
                ),
                "mean_queue_oracle_cost": float(queue_frame["oracle_cost_mean"].mean()),
                "incident_weighted_regret": float(
                    queue_frame["regret_sum"].sum() / total_n
                ),
                "mean_queue_regret": float(queue_frame["regret_mean"].mean()),
                "mean_predicted_risk": float(queue_frame["predicted_robust_risk_mean"].mean()),
                "tp_close_rate": float(queue_frame["tp_close_count"].sum() / tp_total) if tp_total else np.nan,
                "non_fp_close_rate": float(queue_frame["non_fp_close_count"].sum() / non_fp_total) if non_fp_total else np.nan,
                "mean_queue_review_rate": float(queue_frame["review_rate"].mean()),
                "priority_lambda": float(lambda_value)
                if policy == "joint_robust"
                else 0.0,
                "global_review_threshold": float(lambda_value)
                if policy == "global_tp_threshold"
                else np.nan,
                "tv_radius": float(
                    tv_radius
                    if policy
                    in {"robust_no_priority", "joint_robust", "global_tp_threshold"}
                    else 0.0
                ),
                "automation_cost_width": float(
                    automation_cost_width
                    if policy
                    in {"robust_no_priority", "joint_robust", "global_tp_threshold"}
                    else 0.0
                ),
                "close_threshold": float(close_threshold)
                if policy
                in {"robust_no_priority", "joint_robust", "global_tp_threshold"}
                else np.nan,
                "closure_safety": closure_safety
                if policy
                in {"robust_no_priority", "joint_robust", "global_tp_threshold"}
                else "none",
            }
            if has_expert_rank:
                aggregate["mean_expert_top_budget_coverage"] = float(
                    queue_frame["expert_top_budget_coverage"].mean()
                )
                aggregate["mean_expert_precision_at_budget"] = float(
                    queue_frame["expert_precision_at_budget"].mean()
                )
                aggregate["mean_expert_priority_gain_ratio"] = float(
                    queue_frame["expert_priority_gain_ratio"].mean()
                )
                aggregate["mean_expert_ndcg_at_budget"] = float(
                    queue_frame["expert_ndcg_at_budget"].mean()
                )
            aggregate_rows.append(aggregate)

    return (
        pd.DataFrame(per_queue_rows),
        pd.DataFrame(aggregate_rows),
        pd.DataFrame(stress_rows),
    )


def evaluate_global_threshold_baseline(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    priority: np.ndarray,
    scope: str,
    budget_values: Iterable[float],
    exact_budgets: bool,
    thresholds: dict[float, float],
    cost_scenario: str,
    tv_radius: float,
    automation_cost_width: float,
    close_threshold: float,
    closure_safety: str,
    seed: int,
    probability_variant: str,
    evaluate_deployment_stress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    queue_frames: list[pd.DataFrame] = []
    aggregate_frames: list[pd.DataFrame] = []
    stress_frames: list[pd.DataFrame] = []
    for budget_value in budget_values:
        target_rate = (
            float(budget_value) / 20.0 if exact_budgets else float(budget_value)
        )
        threshold = thresholds[round(target_rate, 10)]
        queue, aggregate, stress = evaluate_policy_grid(
            frame,
            probabilities,
            priority,
            scope,
            [budget_value],
            exact_budgets,
            [],
            cost_scenario,
            tv_radius,
            automation_cost_width,
            close_threshold,
            closure_safety,
            seed,
            probability_variant,
            policy_specs_override=[
                ("global_tp_threshold", "global_tp_threshold", threshold)
            ],
            evaluate_deployment_stress=evaluate_deployment_stress,
        )
        queue_frames.append(queue)
        aggregate_frames.append(aggregate)
        stress_frames.append(stress)
    return (
        pd.concat(queue_frames, ignore_index=True),
        pd.concat(aggregate_frames, ignore_index=True),
        pd.concat(stress_frames, ignore_index=True),
    )


def aggregate_deployment_stress(per_queue: pd.DataFrame) -> pd.DataFrame:
    if per_queue.empty:
        return pd.DataFrame()
    keys = [
        "scope",
        "probability_variant",
        "policy",
        "decision_cost_scenario",
        "budget_value",
        "budget_is_exact",
        "budget_allocation_mode",
        "deployment_cost_scenario",
        "deployment_automation_cost_width",
        "cost_scale",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in per_queue.groupby(keys, sort=True, dropna=False):
        values = dict(zip(keys, key))
        incidents = int(group["queue_size"].sum())
        values.update(
            {
                "queues": int(len(group)),
                "incidents": incidents,
                "reviews": int(group["reviews"].sum()),
                "actual_review_rate": float(group["reviews"].sum() / incidents),
                "incident_weighted_cost": float(
                    group["realized_cost_sum"].sum() / incidents
                ),
                "incident_weighted_normalized_cost": float(
                    group["normalized_realized_cost_sum"].sum() / incidents
                ),
                "incident_weighted_oracle_cost": float(
                    group["oracle_cost_sum"].sum() / incidents
                ),
                "incident_weighted_regret": float(
                    group["regret_sum"].sum() / incidents
                ),
                "incident_weighted_normalized_regret": float(
                    group["normalized_regret_sum"].sum() / incidents
                ),
                "mean_queue_cost": float(group["realized_cost_mean"].mean()),
                "mean_queue_regret": float(group["regret_mean"].mean()),
                "macro_queue_normalized_cost": float(
                    group["normalized_realized_cost_mean"].mean()
                ),
                "macro_queue_normalized_regret": float(
                    group["normalized_regret_mean"].mean()
                ),
            }
        )
        rows.append(values)
    return pd.DataFrame(rows)


def summarize_worst_case_stress(aggregate: pd.DataFrame) -> pd.DataFrame:
    if aggregate.empty:
        return pd.DataFrame()
    keys = [
        "scope",
        "probability_variant",
        "policy",
        "decision_cost_scenario",
        "budget_value",
        "budget_is_exact",
        "budget_allocation_mode",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in aggregate.groupby(keys, sort=True, dropna=False):
        values = dict(zip(keys, key))
        macro_cost_row = group.loc[group["macro_queue_normalized_cost"].idxmax()]
        macro_regret_row = group.loc[group["macro_queue_normalized_regret"].idxmax()]
        weighted_cost_row = group.loc[
            group["incident_weighted_normalized_cost"].idxmax()
        ]
        weighted_regret_row = group.loc[
            group["incident_weighted_normalized_regret"].idxmax()
        ]
        values.update(
            {
                "worst_case_macro_queue_normalized_cost": float(
                    macro_cost_row["macro_queue_normalized_cost"]
                ),
                "worst_macro_cost_deployment_scenario": macro_cost_row[
                    "deployment_cost_scenario"
                ],
                "worst_macro_cost_automation_width": float(
                    macro_cost_row["deployment_automation_cost_width"]
                ),
                "worst_case_macro_queue_normalized_regret": float(
                    macro_regret_row["macro_queue_normalized_regret"]
                ),
                "worst_macro_regret_deployment_scenario": macro_regret_row[
                    "deployment_cost_scenario"
                ],
                "worst_macro_regret_automation_width": float(
                    macro_regret_row["deployment_automation_cost_width"]
                ),
                "worst_case_incident_weighted_normalized_cost": float(
                    weighted_cost_row["incident_weighted_normalized_cost"]
                ),
                "worst_weighted_cost_deployment_scenario": weighted_cost_row[
                    "deployment_cost_scenario"
                ],
                "worst_weighted_cost_automation_width": float(
                    weighted_cost_row["deployment_automation_cost_width"]
                ),
                "worst_case_incident_weighted_normalized_regret": float(
                    weighted_regret_row["incident_weighted_normalized_regret"]
                ),
                "worst_weighted_regret_deployment_scenario": weighted_regret_row[
                    "deployment_cost_scenario"
                ],
                "worst_weighted_regret_automation_width": float(
                    weighted_regret_row["deployment_automation_cost_width"]
                ),
            }
        )
        rows.append(values)
    return pd.DataFrame(rows)


def queue_size_strata(per_queue: pd.DataFrame) -> pd.DataFrame:
    if per_queue.empty:
        return pd.DataFrame()
    work = per_queue.copy()
    work["queue_size_band"] = pd.cut(
        work["queue_size"],
        bins=QUEUE_SIZE_BINS,
        labels=QUEUE_SIZE_LABELS,
        right=True,
    ).astype("string")
    keys = [
        "scope",
        "probability_variant",
        "policy",
        "cost_scenario",
        "budget_value",
        "budget_is_exact",
        "budget_allocation_mode",
        "queue_size_band",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in work.groupby(keys, sort=True, observed=True, dropna=False):
        values = dict(zip(keys, key))
        incidents = int(group["queue_size"].sum())
        tp_count = int(group["tp_count"].sum())
        non_fp_count = int(group["non_fp_count"].sum())
        values.update(
            {
                "queues": int(len(group)),
                "incidents": incidents,
                "actual_review_rate": float(group["reviews"].sum() / incidents),
                "incident_weighted_cost": float(
                    group["realized_cost_sum"].sum() / incidents
                ),
                "incident_weighted_regret": float(
                    group["regret_sum"].sum() / incidents
                ),
                "tp_close_rate": float(group["tp_close_count"].sum() / tp_count)
                if tp_count
                else np.nan,
                "non_fp_close_rate": float(
                    group["non_fp_close_count"].sum() / non_fp_count
                )
                if non_fp_count
                else np.nan,
                "zero_review_budget_rate": float(
                    group["review_budget"].eq(0).mean()
                ),
            }
        )
        for metric in (
            "expert_top_budget_coverage",
            "expert_precision_at_budget",
            "expert_priority_gain_ratio",
            "expert_ndcg_at_budget",
        ):
            if metric in group.columns and group[metric].notna().any():
                values[f"mean_{metric}"] = float(group[metric].mean())
        rows.append(values)
    return pd.DataFrame(rows)


def paired_cluster_bootstrap(
    per_queue: pd.DataFrame,
    proposed: str,
    comparator: str,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Paired organization-cluster bootstrap for cost, safety, and ranking.

    Ratio estimands are recomputed from their numerators and denominators in
    every bootstrap replicate. This yields intervals that correspond to the
    aggregate incident-weighted cost and closure-rate columns. Equal-queue
    mean cost and expert-ranking metrics remain queue-average estimands.
    """

    base_columns = [
        "organization",
        "queue_size",
        "realized_cost_mean",
        "realized_cost_sum",
        "regret_mean",
        "regret_sum",
        "tp_close_count",
        "tp_count",
        "non_fp_close_count",
        "non_fp_count",
    ]
    expert_columns = [
        "expert_top_budget_coverage",
        "expert_precision_at_budget",
        "expert_priority_gain_ratio",
        "expert_ndcg_at_budget",
    ]
    available_expert = [
        column
        for column in expert_columns
        if column in per_queue.columns and per_queue[column].notna().any()
    ]
    columns = base_columns + available_expert
    left = per_queue[per_queue["policy"] == proposed][columns]
    right = per_queue[per_queue["policy"] == comparator][columns]
    merged = left.merge(
        right,
        on="organization",
        how="inner",
        validate="one_to_one",
        suffixes=("_proposed", "_comparator"),
    )
    if merged.empty:
        return []
    n = len(merged)
    for denominator in ("queue_size", "tp_count", "non_fp_count"):
        if not np.array_equal(
            merged[f"{denominator}_proposed"].to_numpy(),
            merged[f"{denominator}_comparator"].to_numpy(),
        ):
            raise RuntimeError(
                f"Paired policies disagree on {denominator}: {proposed} vs {comparator}"
            )

    simple_differences: dict[str, np.ndarray] = {
        "mean_queue_cost": (
            merged["realized_cost_mean_proposed"]
            - merged["realized_cost_mean_comparator"]
        ).to_numpy(dtype=float),
        "mean_queue_regret": (
            merged["regret_mean_proposed"]
            - merged["regret_mean_comparator"]
        ).to_numpy(dtype=float),
    }
    for metric in available_expert:
        valid = merged[
            [f"{metric}_proposed", f"{metric}_comparator"]
        ].notna().all(axis=1)
        if valid.all():
            simple_differences[metric] = (
                merged[f"{metric}_proposed"] - merged[f"{metric}_comparator"]
            ).to_numpy(dtype=float)

    ratio_specs = {
        "incident_weighted_cost": (
            "realized_cost_sum_proposed",
            "queue_size_proposed",
            "realized_cost_sum_comparator",
            "queue_size_comparator",
        ),
        "incident_weighted_regret": (
            "regret_sum_proposed",
            "queue_size_proposed",
            "regret_sum_comparator",
            "queue_size_comparator",
        ),
        "tp_close_rate": (
            "tp_close_count_proposed",
            "tp_count_proposed",
            "tp_close_count_comparator",
            "tp_count_comparator",
        ),
        "non_fp_close_rate": (
            "non_fp_close_count_proposed",
            "non_fp_count_proposed",
            "non_fp_close_count_comparator",
            "non_fp_count_comparator",
        ),
    }
    ratio_arrays = {
        metric: tuple(merged[column].to_numpy(dtype=float) for column in columns)
        for metric, columns in ratio_specs.items()
    }

    estimates = {
        metric: np.empty(repetitions, dtype=float)
        for metric in [*simple_differences, *ratio_arrays]
    }
    rng = np.random.default_rng(seed)
    batch = max(1, min(200, repetitions))
    cursor = 0
    while cursor < repetitions:
        size = min(batch, repetitions - cursor)
        indices = rng.integers(0, n, size=(size, n))
        for metric, difference in simple_differences.items():
            estimates[metric][cursor : cursor + size] = difference[indices].mean(axis=1)
        for metric, arrays in ratio_arrays.items():
            left_num, left_den, right_num, right_den = arrays
            left_denominator = left_den[indices].sum(axis=1)
            right_denominator = right_den[indices].sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                estimates[metric][cursor : cursor + size] = (
                    left_num[indices].sum(axis=1) / left_denominator
                    - right_num[indices].sum(axis=1) / right_denominator
                )
        cursor += size

    points: dict[str, float] = {
        metric: float(difference.mean())
        for metric, difference in simple_differences.items()
    }
    for metric, arrays in ratio_arrays.items():
        left_num, left_den, right_num, right_den = arrays
        if left_den.sum() <= 0 or right_den.sum() <= 0:
            continue
        points[metric] = float(
            left_num.sum() / left_den.sum() - right_num.sum() / right_den.sum()
        )

    rows: list[dict[str, Any]] = []
    for metric, point in points.items():
        finite_estimates = estimates[metric][np.isfinite(estimates[metric])]
        if len(finite_estimates) != repetitions:
            raise RuntimeError(f"Non-finite bootstrap estimates for {metric}")
        rows.append(
            {
                "proposed": proposed,
                "comparator": comparator,
                "metric": metric,
                "queues": n,
                "mean_difference": point,
                "ci_lower": float(np.quantile(finite_estimates, 0.025)),
                "ci_upper": float(np.quantile(finite_estimates, 0.975)),
                "probability_difference_below_zero": float(
                    np.mean(finite_estimates < 0)
                ),
                "probability_difference_above_zero": float(
                    np.mean(finite_estimates > 0)
                ),
                "bootstrap_p_two_sided": float(
                    min(
                        1.0,
                        2.0
                        * (
                            min(
                                int(np.sum(finite_estimates <= 0)),
                                int(np.sum(finite_estimates >= 0)),
                            )
                            + 1
                        )
                        / (len(finite_estimates) + 1),
                    )
                ),
            }
        )
    return rows


def add_holm_adjustment(
    frame: pd.DataFrame, family_columns: list[str]
) -> pd.DataFrame:
    """Apply Holm family-wise error correction to bootstrap sign p-values."""
    if frame.empty:
        return frame
    result = frame.copy()
    result["holm_adjusted_p"] = np.nan
    for _key, group in result.groupby(family_columns, sort=False, dropna=False):
        ordered = group["bootstrap_p_two_sided"].sort_values().index.to_numpy()
        raw = result.loc[ordered, "bootstrap_p_two_sided"].to_numpy(dtype=float)
        multipliers = np.arange(len(raw), 0, -1, dtype=float)
        adjusted = np.maximum.accumulate(raw * multipliers)
        result.loc[ordered, "holm_adjusted_p"] = np.minimum(1.0, adjusted)
    result["significant_holm_0_05"] = result["holm_adjusted_p"] < 0.05
    return result


def run_bootstraps(
    per_queue: pd.DataFrame,
    primary_lambda: float,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    proposed = f"joint_robust_lambda_{primary_lambda:g}"
    rows: list[dict[str, Any]] = []
    grouped = per_queue.groupby(
        [
            "scope",
            "probability_variant",
            "cost_scenario",
            "budget_value",
            "budget_is_exact",
        ],
        sort=True,
    )
    for number, (key, group) in enumerate(grouped):
        scope, probability_variant, cost_scenario, budget_value, exact = key
        for comparator in (
            "nominal_optimizer",
            "robust_no_priority",
            "priority_only",
            "priority_defer_only",
            "tp_probability",
            "uncertainty",
            "alert_count",
            "arrival_oldest",
            "random",
        ):
            results = paired_cluster_bootstrap(
                group,
                proposed,
                comparator,
                repetitions,
                seed + number * 100 + len(rows),
            )
            for result in results:
                result.update(
                    {
                        "scope": scope,
                        "probability_variant": probability_variant,
                        "cost_scenario": cost_scenario,
                        "budget_value": float(budget_value),
                        "budget_is_exact": bool(exact),
                    }
                )
                rows.append(result)
    return add_holm_adjustment(
        pd.DataFrame(rows),
        ["scope", "probability_variant", "cost_scenario", "metric"],
    )


def run_probability_ablation_bootstraps(
    per_queue: pd.DataFrame,
    primary_lambda: float,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    """Compare temperature-scaled policies with identical raw-probability policies."""
    proposed = f"joint_robust_lambda_{primary_lambda:g}"
    policies = (proposed, "robust_no_priority", "nominal_optimizer")
    rows: list[dict[str, Any]] = []
    grouping = ["scope", "cost_scenario", "budget_value", "budget_is_exact"]
    for number, (key, group) in enumerate(per_queue.groupby(grouping, sort=True)):
        scope, cost_scenario, budget_value, exact = key
        for policy in policies:
            work = group[group["policy"].eq(policy)].copy()
            if set(work["probability_variant"].unique()) != {"raw", "temperature"}:
                continue
            work["base_policy"] = work["policy"]
            work["policy"] = work["probability_variant"].map(
                {"temperature": "temperature_policy", "raw": "raw_policy"}
            )
            results = paired_cluster_bootstrap(
                work,
                "temperature_policy",
                "raw_policy",
                repetitions,
                seed + number * 100 + len(rows),
            )
            for result in results:
                result.update(
                    {
                        "scope": scope,
                        "policy": policy,
                        "cost_scenario": cost_scenario,
                        "budget_value": float(budget_value),
                        "budget_is_exact": bool(exact),
                        "difference_definition": "temperature_minus_raw",
                    }
                )
                rows.append(result)
    return add_holm_adjustment(
        pd.DataFrame(rows), ["scope", "policy", "cost_scenario", "metric"]
    )


def paired_worst_case_stress_bootstrap(
    per_queue: pd.DataFrame,
    proposed: str,
    comparator: str,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Bootstrap policy differences under each policy's worst condition.

    Macro queue estimands are primary, matching the research charter. The
    incident-weighted estimands are retained explicitly as secondary results.
    """
    condition = ["deployment_cost_scenario", "deployment_automation_cost_width"]
    columns = [
        "organization",
        *condition,
        "queue_size",
        "normalized_realized_cost_sum",
        "normalized_realized_cost_mean",
        "normalized_regret_sum",
        "normalized_regret_mean",
    ]
    left = per_queue[per_queue["policy"].eq(proposed)][columns]
    right = per_queue[per_queue["policy"].eq(comparator)][columns]
    merged = left.merge(
        right,
        on=["organization", *condition],
        validate="one_to_one",
        suffixes=("_proposed", "_comparator"),
    )
    if merged.empty:
        return []
    if not np.array_equal(
        merged["queue_size_proposed"].to_numpy(),
        merged["queue_size_comparator"].to_numpy(),
    ):
        raise RuntimeError("Stress bootstrap queue sizes differ between policies")
    organizations = sorted(merged["organization"].unique())
    conditions = sorted(
        set(zip(merged[condition[0]], merged[condition[1]])), key=lambda x: (x[0], x[1])
    )

    def matrix(column: str) -> np.ndarray:
        pivot = merged.pivot(
            index="organization", columns=condition, values=column
        ).reindex(index=organizations, columns=pd.MultiIndex.from_tuples(conditions))
        values = pivot.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Incomplete deployment stress matrix for {column}")
        return values

    queue_size = (
        merged.drop_duplicates("organization")
        .set_index("organization")
        .reindex(organizations)["queue_size_proposed"]
        .to_numpy(dtype=float)
    )
    arrays = {
        "worst_case_macro_queue_normalized_cost": (
            matrix("normalized_realized_cost_mean_proposed"),
            matrix("normalized_realized_cost_mean_comparator"),
            "macro",
        ),
        "worst_case_macro_queue_normalized_regret": (
            matrix("normalized_regret_mean_proposed"),
            matrix("normalized_regret_mean_comparator"),
            "macro",
        ),
        "worst_case_incident_weighted_normalized_cost": (
            matrix("normalized_realized_cost_sum_proposed"),
            matrix("normalized_realized_cost_sum_comparator"),
            "incident_weighted",
        ),
        "worst_case_incident_weighted_normalized_regret": (
            matrix("normalized_regret_sum_proposed"),
            matrix("normalized_regret_sum_comparator"),
            "incident_weighted",
        ),
    }
    n = len(organizations)
    estimates = {metric: np.empty(repetitions) for metric in arrays}
    rng = np.random.default_rng(seed)
    batch = max(1, min(100, repetitions))
    cursor = 0
    while cursor < repetitions:
        size = min(batch, repetitions - cursor)
        indices = rng.integers(0, n, size=(size, n))
        denominators = queue_size[indices].sum(axis=1)[:, None]
        for metric, (left_values, right_values, estimand) in arrays.items():
            if estimand == "macro":
                left_worst = left_values[indices].mean(axis=1).max(axis=1)
                right_worst = right_values[indices].mean(axis=1).max(axis=1)
            else:
                left_worst = (
                    left_values[indices].sum(axis=1) / denominators
                ).max(axis=1)
                right_worst = (
                    right_values[indices].sum(axis=1) / denominators
                ).max(axis=1)
            estimates[metric][cursor : cursor + size] = left_worst - right_worst
        cursor += size

    rows: list[dict[str, Any]] = []
    denominator = queue_size.sum()
    for metric, (left_values, right_values, estimand) in arrays.items():
        if estimand == "macro":
            point = float(
                left_values.mean(axis=0).max()
                - right_values.mean(axis=0).max()
            )
        else:
            point = float(
                (left_values.sum(axis=0) / denominator).max()
                - (right_values.sum(axis=0) / denominator).max()
            )
        values = estimates[metric]
        rows.append(
            {
                "proposed": proposed,
                "comparator": comparator,
                "metric": metric,
                "queues": n,
                "mean_difference": point,
                "ci_lower": float(np.quantile(values, 0.025)),
                "ci_upper": float(np.quantile(values, 0.975)),
                "probability_difference_below_zero": float(np.mean(values < 0)),
                "probability_difference_above_zero": float(np.mean(values > 0)),
                "bootstrap_p_two_sided": float(
                    min(
                        1.0,
                        2.0
                        * (min(int(np.sum(values <= 0)), int(np.sum(values >= 0))) + 1)
                        / (len(values) + 1),
                    )
                ),
            }
        )
    return rows


def run_worst_case_stress_bootstraps(
    per_queue: pd.DataFrame,
    primary_lambda: float,
    selected_probability_variant: str,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    proposed = f"joint_robust_lambda_{primary_lambda:g}"
    selected = per_queue[
        per_queue["probability_variant"].eq(selected_probability_variant)
    ]
    rows: list[dict[str, Any]] = []
    grouping = [
        "scope",
        "probability_variant",
        "decision_cost_scenario",
        "budget_value",
        "budget_is_exact",
    ]
    for number, (key, group) in enumerate(selected.groupby(grouping, sort=True)):
        scope, variant, decision_scenario, budget_value, exact = key
        for comparator in (
            "nominal_optimizer",
            "robust_no_priority",
            "priority_only",
            "priority_defer_only",
        ):
            results = paired_worst_case_stress_bootstrap(
                group,
                proposed,
                comparator,
                repetitions,
                seed + number * 100 + len(rows),
            )
            for result in results:
                result.update(
                    {
                        "scope": scope,
                        "probability_variant": variant,
                        "decision_cost_scenario": decision_scenario,
                        "budget_value": float(budget_value),
                        "budget_is_exact": bool(exact),
                    }
                )
                rows.append(result)
    return add_holm_adjustment(
        pd.DataFrame(rows),
        ["scope", "probability_variant", "decision_cost_scenario", "metric"],
    )


def sensitivity_analysis(
    ranked: pd.DataFrame,
    probabilities: np.ndarray,
    priority: np.ndarray,
    budgets: list[int],
    primary_lambda: float,
    seed: int,
    probability_variant: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for scenario in COST_SCENARIOS:
        for radius in (0.0, 0.025, 0.05, 0.10):
            for width in (0.0, 0.25, 0.50):
                for threshold in (0.05, 0.10, 0.20):
                    for closure_safety in ("tp", "tp_bp"):
                        _queue, aggregate, _stress = evaluate_policy_grid(
                            ranked,
                            probabilities,
                            priority,
                            scope="ranked_sensitivity",
                            budget_values=budgets,
                            exact_budgets=True,
                            priority_lambdas=[primary_lambda],
                            cost_scenario=scenario,
                            tv_radius=radius,
                            automation_cost_width=width,
                            close_threshold=threshold,
                            closure_safety=closure_safety,
                            seed=seed,
                            probability_variant=probability_variant,
                            policy_specs_override=[
                                ("nominal_optimizer", "nominal_optimizer", 0.0),
                                ("robust_no_priority", "robust_no_priority", 0.0),
                                (
                                    "priority_defer_only",
                                    "priority_defer_only",
                                    0.0,
                                ),
                                (
                                    f"joint_robust_lambda_{primary_lambda:g}",
                                    "joint_robust",
                                    primary_lambda,
                                ),
                            ],
                        )
                        keep = aggregate[
                            aggregate["policy"].isin(
                                [
                                    "nominal_optimizer",
                                    "robust_no_priority",
                                    "priority_defer_only",
                                    f"joint_robust_lambda_{primary_lambda:g}",
                                ]
                            )
                        ].copy()
                        keep["sensitivity_tv_radius"] = radius
                        keep["sensitivity_automation_cost_width"] = width
                        keep["sensitivity_close_threshold"] = threshold
                        keep["sensitivity_closure_safety"] = closure_safety
                        rows.append(keep)
    return pd.concat(rows, ignore_index=True)


def largest_remainder_budget_sensitivity(
    test: pd.DataFrame,
    probabilities: np.ndarray,
    builder: IncidentFeatureBuilder,
    budgets: list[float],
    primary_lambda: float,
    tv_radius: float,
    automation_cost_width: float,
    close_threshold: float,
    closure_safety: str,
    seed: int,
    probability_variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a pooled-capacity largest-remainder allocation sensitivity."""
    proposed = f"joint_robust_lambda_{primary_lambda:g}"
    policy_specs = [
        ("nominal_optimizer", "nominal_optimizer", 0.0),
        ("robust_no_priority", "robust_no_priority", 0.0),
        ("priority_defer_only", "priority_defer_only", 0.0),
        (proposed, "joint_robust", primary_lambda),
    ]
    queue_outputs: list[pd.DataFrame] = []
    aggregate_outputs: list[pd.DataFrame] = []
    for scope, mask in (
        ("public", test[EVALUATION_ONLY].eq("Public").to_numpy()),
        ("private", test[EVALUATION_ONLY].eq("Private").to_numpy()),
    ):
        scoped = test.loc[mask].reset_index(drop=True)
        scoped_priority = builder.priority_score(scoped).to_numpy(dtype=float)
        queue_frame, aggregate_frame, _stress = evaluate_policy_grid(
            scoped,
            probabilities[mask],
            scoped_priority,
            scope,
            budgets,
            False,
            [primary_lambda],
            "medium",
            tv_radius,
            automation_cost_width,
            close_threshold,
            closure_safety,
            seed,
            probability_variant,
            policy_specs_override=policy_specs,
            budget_allocation_mode="largest_remainder",
        )
        queue_outputs.append(queue_frame)
        aggregate_outputs.append(aggregate_frame)
    return (
        pd.concat(queue_outputs, ignore_index=True),
        pd.concat(aggregate_outputs, ignore_index=True),
    )


def save_predictions(
    path: Path,
    frame: pd.DataFrame,
    probability_variants: dict[str, np.ndarray],
    selected_variant: str,
    priority: np.ndarray,
) -> None:
    output = frame[KEY_COLUMNS + [TARGET, EVALUATION_ONLY, "first_timestamp", "alert_count", "evidence_rows"]].copy()
    for variant, probabilities in probability_variants.items():
        for index, label in enumerate(LABELS):
            output[f"p_{variant}_{label}"] = probabilities[:, index]
    output["selected_probability_variant"] = selected_variant
    output["priority_score"] = priority
    connection = duckdb.connect(database=":memory:")
    connection.register("prediction_frame", output)
    escaped = str(path.resolve()).replace("'", "''")
    connection.execute(
        f"COPY prediction_frame TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    connection.close()


def plot_results(
    aggregate: pd.DataFrame,
    bootstrap: pd.DataFrame,
    probability_bootstrap: pd.DataFrame,
    stress_bootstrap: pd.DataFrame,
    output: Path,
    primary_lambda: float,
    selected_probability_variant: str,
) -> None:
    proposed = f"joint_robust_lambda_{primary_lambda:g}"
    scope_labels = {"public": "Public", "private": "Private"}

    def effect_plot(
        frame: pd.DataFrame,
        path_stem: str,
        xlabel: str,
        ylabel: str,
        scale: float = 100.0,
        exact_budget: bool = False,
    ) -> None:
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        for scope, group in frame.groupby("scope", sort=True):
            group = group.sort_values("budget_value")
            x = group["budget_value"].to_numpy(dtype=float)
            if not exact_budget:
                x = x * 100.0
            point = group["mean_difference"].to_numpy(dtype=float) * scale
            lower = group["ci_lower"].to_numpy(dtype=float) * scale
            upper = group["ci_upper"].to_numpy(dtype=float) * scale
            ax.errorbar(
                x,
                point,
                yerr=np.vstack([point - lower, upper - point]),
                marker="o",
                capsize=3,
                linewidth=1.4,
                label=scope_labels.get(str(scope), str(scope).title()),
            )
        ax.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(output / f"{path_stem}.png", dpi=300)
        fig.savefig(output / f"{path_stem}.pdf")
        plt.close(fig)

    cost_effect = stress_bootstrap[
        stress_bootstrap["proposed"].eq(proposed)
        & stress_bootstrap["comparator"].eq("nominal_optimizer")
        & stress_bootstrap["metric"].eq(
            "worst_case_macro_queue_normalized_cost"
        )
        & stress_bootstrap["scope"].isin(["public", "private"])
    ]
    effect_plot(
        cost_effect,
        "figure_1_worst_case_cost_effect",
        "Analyst review budget (%)",
        "Worst-case macro normalized cost difference\n(joint - nominal; percentage points)",
    )

    priority_effect = bootstrap[
        bootstrap["proposed"].eq(proposed)
        & bootstrap["comparator"].eq("tp_probability")
        & bootstrap["metric"].eq("expert_top_budget_coverage")
        & bootstrap["scope"].eq("ranked")
        & bootstrap["probability_variant"].eq(selected_probability_variant)
    ]
    effect_plot(
        priority_effect,
        "figure_2_priority_coverage_effect",
        "Review budget per 20-incident expert queue",
        "Expert top-budget coverage difference\n(joint - TP probability; percentage points)",
        exact_budget=True,
    )

    calibration_effect = probability_bootstrap[
        probability_bootstrap["policy"].eq(proposed)
        & probability_bootstrap["metric"].eq("tp_close_rate")
        & probability_bootstrap["scope"].isin(["public", "private"])
    ]
    effect_plot(
        calibration_effect,
        "figure_3_calibration_safety_effect",
        "Analyst review budget (%)",
        "TP automatic-closure difference\n(temperature - raw; percentage points)",
    )

    # Descriptive supplement. Unlike the main effect figures, this is not an
    # uncertainty plot and the cost is explicitly labeled in scenario units.
    public = aggregate[
        aggregate["scope"].eq("public")
        & aggregate["policy"].isin(
            ["nominal_optimizer", "robust_no_priority", proposed]
        )
        & aggregate["probability_variant"].eq(selected_probability_variant)
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for policy, group in public.groupby("policy"):
        group = group.sort_values("actual_review_rate")
        ax.plot(
            group["actual_review_rate"] * 100,
            group["incident_weighted_cost"],
            marker="o",
            label=policy.replace("_", " "),
        )
    ax.set_xlabel("Actual analyst review rate (%)")
    ax.set_ylabel("Realized cost per incident (scenario units)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "figure_s1_cost_vs_budget.png", dpi=300)
    fig.savefig(output / "figure_s1_cost_vs_budget.pdf")
    plt.close(fig)


def write_protocol_deviations(output: Path) -> None:
    rows = [
        {
            "protocol_item": "Raw-versus-calibrated policy ablation",
            "status": "resolved_in_v0.5_and_retained",
            "handling": "Both raw and temperature-scaled HGB probabilities are evaluated under identical policies and compared by paired organization bootstrap.",
        },
        {
            "protocol_item": "Oracle regret",
            "status": "resolved_in_v0.5",
            "handling": "Regret uses an unattainable label-aware oracle with the same at-most review capacity and each declared deployment cost matrix.",
        },
        {
            "protocol_item": "Deployment-cost shift",
            "status": "resolved_in_v0.5",
            "handling": "Frozen actions are evaluated under low, medium, and high matrices and automation-cost widths 0, 0.25, and 0.50.",
        },
        {
            "protocol_item": "Independent threshold baseline",
            "status": "resolved_in_v0.5",
            "handling": "Global TP-probability thresholds are frozen on calibration-selection organizations; queue-capacity violations are reported rather than hidden.",
        },
        {
            "protocol_item": "Multiple-comparison control",
            "status": "resolved_in_v0.5",
            "handling": "Holm-adjusted bootstrap sign-test p-values are reported within each scope, probability variant, cost scenario, and metric family.",
        },
        {
            "protocol_item": "Exact public AIP TF-IDF/BM25 variants",
            "status": "not_implemented",
            "handling": "The incident Parquet retains aggregate counts and first-observed components, not the full component representation required for a faithful AIP reproduction. The queue-relative rarity score is labeled as a heuristic, not as AIP.",
        },
        {
            "protocol_item": "Maximum product-severity baseline",
            "status": "not_implemented",
            "handling": "The validated incident schema does not contain a product-severity variable. Alert-count, TP-probability, entropy, arrival, random, priority-only, and optimization baselines are reported instead.",
        },
        {
            "protocol_item": "Validation-selected closure threshold",
            "status": "deviation",
            "handling": "The primary threshold 0.10 was prespecified rather than optimized; thresholds 0.05, 0.10, and 0.20 are reported in sensitivity analysis. No claim of validation-optimality is permitted.",
        },
        {
            "protocol_item": "NDCG and precision-at-K",
            "status": "resolved_in_v0.6",
            "handling": "The ranked queues now report precision at the exact review budget and NDCG using relevance 21 minus expert QueueRank; policy scores order the reviewed set and expert rank is evaluation-only.",
        },
        {
            "protocol_item": "Macro-queue primary estimand",
            "status": "corrected_in_v0.6",
            "handling": "Worst-case normalized cost and regret now use equal-weight organization-queue means as primary estimands. Incident-weighted versions remain explicitly labeled secondary results.",
        },
        {
            "protocol_item": "No triage-probability ablation",
            "status": "resolved_in_v0.6",
            "handling": "priority_defer_only selects reviews solely from the queue-relative priority score and defers every unreviewed incident; no incident probability affects its actions.",
        },
        {
            "protocol_item": "Fractional budget allocation",
            "status": "sensitivity_added_in_v0.6",
            "handling": "The preregistered per-queue floor rule remains primary. A deterministic largest-remainder sensitivity preserves the pooled system target and quantifies zero-budget queues by queue-size stratum.",
        },
        {
            "protocol_item": "Uncertainty figures",
            "status": "corrected_in_v0.6",
            "handling": "The three main figures display paired organization-cluster bootstrap effects with 95% intervals. Raw cost curves are supplementary and labeled in scenario units.",
        },
        {
            "protocol_item": "Stakeholder-validated monetary costs",
            "status": "not_available",
            "handling": "All costs remain normalized scenario units. Dollar savings, ROI, and deployment economics are prohibited claims.",
        },
    ]
    pd.DataFrame(rows).to_csv(output / "protocol_deviations.csv", index=False)


def make_upload_bundle(output: Path) -> Path:
    bundle = output / "guide_results_bundle.zip"
    allowed_suffixes = {".csv", ".json", ".png", ".pdf", ".txt", ".gz"}
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path == bundle or path.suffix.lower() not in allowed_suffixes:
                continue
            archive.write(path, arcname=path.name)
    return bundle


def main() -> None:
    args = parse_args()
    started = time.time()
    stage_timings: dict[str, float] = {}
    if args.primary_lambda not in args.priority_lambdas:
        raise ValueError("--primary-lambda must appear in --priority-lambdas")
    if not 0 < args.calibration_fraction < 0.5:
        raise ValueError("--calibration-fraction must be between 0 and 0.5")
    if not 0 < args.calibration_selection_fraction < 1:
        raise ValueError("--calibration-selection-fraction must be between 0 and 1")
    if args.automation_cost_width < 0:
        raise ValueError("--automation-cost-width must be nonnegative")
    build_report = verify_inputs(args)
    prepare_output(args.output_dir, args.overwrite)

    stage_started = time.time()
    print("Loading incident Parquet files...", flush=True)
    full_train = load_incidents(args.train_parquet, is_test=False)
    full_test = load_incidents(args.test_parquet, is_test=True)
    if len(full_train) != int(build_report["train"]["incident_count"]):
        raise RuntimeError("Training incident count differs from build report")
    if len(full_test) != int(build_report["test"]["incident_count"]):
        raise RuntimeError("Test incident count differs from build report")
    stage_timings["load_incident_parquet"] = time.time() - stage_started

    unlabeled = int(full_train[TARGET].isna().sum())
    train = full_train[full_train[TARGET].isin(LABELS)].copy().reset_index(drop=True)
    test = full_test[full_test[TARGET].isin(LABELS)].copy().reset_index(drop=True)
    train = stratified_limit(train, args.max_train_incidents, args.seed)
    test = stratified_limit(test, args.max_test_incidents, args.seed + 1)
    if set(train[TARGET].dropna().unique()) != set(LABELS) or set(test[TARGET].dropna().unique()) != set(LABELS):
        raise RuntimeError("Every modeling split must contain TP, BP, and FP")

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=args.calibration_fraction, random_state=args.seed
    )
    fit_index, calibration_index = next(
        splitter.split(train, train[TARGET], groups=train["OrgId"])
    )
    fit = train.iloc[fit_index].reset_index(drop=True)
    calibration = train.iloc[calibration_index].reset_index(drop=True)
    if set(fit["OrgId"]) & set(calibration["OrgId"]):
        raise RuntimeError("Organization leakage between fit and calibration partitions")
    if set(fit[TARGET].unique()) != set(LABELS) or set(calibration[TARGET].unique()) != set(LABELS):
        raise RuntimeError("Organization split lost a target class")

    calibration_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.calibration_selection_fraction,
        random_state=args.seed + 1,
    )
    calibration_fit_index, calibration_selection_index = next(
        calibration_splitter.split(
            calibration, calibration[TARGET], groups=calibration["OrgId"]
        )
    )
    calibration_fit = calibration.iloc[calibration_fit_index].reset_index(drop=True)
    calibration_selection = calibration.iloc[calibration_selection_index].reset_index(drop=True)
    if set(calibration_fit["OrgId"]) & set(calibration_selection["OrgId"]):
        raise RuntimeError(
            "Organization leakage between calibration-fit and calibration-selection partitions"
        )
    if set(fit["OrgId"]) & (
        set(calibration_fit["OrgId"]) | set(calibration_selection["OrgId"])
    ):
        raise RuntimeError("Organization leakage across model and calibration partitions")
    if (
        set(calibration_fit[TARGET].unique()) != set(LABELS)
        or set(calibration_selection[TARGET].unique()) != set(LABELS)
    ):
        raise RuntimeError("Calibration subdivision lost a target class")

    y_fit = fit[TARGET].map(LABEL_TO_ID).to_numpy(dtype=np.int8)
    y_calibration = calibration[TARGET].map(LABEL_TO_ID).to_numpy(dtype=np.int8)
    y_calibration_fit = y_calibration[calibration_fit_index]
    y_calibration_selection = y_calibration[calibration_selection_index]
    y_test = test[TARGET].map(LABEL_TO_ID).to_numpy(dtype=np.int8)
    builder = IncidentFeatureBuilder().fit(fit)
    print("Constructing leakage-safe model matrices...", flush=True)
    x_fit = builder.transform_hgb(fit)
    x_cal = builder.transform_hgb(calibration)
    x_test = builder.transform_hgb(test)
    categorical_mask = builder.categorical_mask.copy()

    metric_rows: list[dict[str, Any]] = []
    prior = DummyClassifier(strategy="prior").fit(np.zeros((len(y_fit), 1)), y_fit)
    dummy_prob = prior.predict_proba(np.zeros((len(y_test), 1)))
    metric_rows.append(classification_metrics(dummy_prob, y_test, "class_prior", "test_all"))

    linear_metadata: dict[str, Any] | None = None
    if not args.skip_linear:
        print("Training linear SGD baseline...", flush=True)
        lx_fit, lx_cal, lx_test, linear_metadata = builder.linear_matrices(fit, calibration, test)
        linear = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-3,
            max_iter=60,
            tol=1e-4,
            average=True,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=5,
            random_state=args.seed,
        )
        with threadpool_limits(limits=args.threads):
            linear.fit(lx_fit, y_fit)
        linear_raw_cal = stable_ovr_probabilities(linear, lx_cal)
        linear_raw_test = stable_ovr_probabilities(linear, lx_test)
        linear_choice = select_probability_variant(
            linear_raw_cal[calibration_fit_index],
            y_calibration_fit,
            linear_raw_cal[calibration_selection_index],
            y_calibration_selection,
            linear_raw_test,
        )
        metric_rows.extend(
            [
                classification_metrics(
                    linear_raw_cal[calibration_selection_index],
                    y_calibration_selection,
                    "linear_sgd_raw",
                    "calibration_selection_unseen_org",
                ),
                classification_metrics(
                    linear_choice.temperature_selection,
                    y_calibration_selection,
                    "linear_sgd_temperature",
                    "calibration_selection_unseen_org",
                ),
                classification_metrics(
                    linear_raw_test, y_test, "linear_sgd_raw", "test_all"
                ),
                classification_metrics(
                    linear_choice.temperature_test,
                    y_test,
                    "linear_sgd_temperature",
                    "test_all",
                ),
                classification_metrics(
                    linear_choice.selected_test,
                    y_test,
                    "linear_sgd_selected",
                    "test_all",
                ),
            ]
        )
        linear_metadata.update(
            {
                "temperature": linear_choice.scaler.temperature,
                "calibration_fit_log_loss_before": linear_choice.scaler.calibration_loss_before,
                "calibration_fit_log_loss_after": linear_choice.scaler.calibration_loss_after,
                "selection_log_loss_raw": linear_choice.selection_log_loss_raw,
                "selection_log_loss_temperature": linear_choice.selection_log_loss_temperature,
                "selected_probability_variant": linear_choice.selected_variant,
            }
        )
        del lx_fit, lx_cal, lx_test, linear

    print("Training histogram gradient-boosting model...", flush=True)
    hgb_training_started = time.time()
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.08,
        max_iter=args.hgb_iterations,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        max_bins=255,
        categorical_features=categorical_mask,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=20,
        random_state=args.seed,
    )
    with threadpool_limits(limits=args.threads):
        model.fit(x_fit, y_fit)
        stage_timings["hgb_training"] = time.time() - hgb_training_started
        hgb_inference_started = time.time()
        raw_cal = model.predict_proba(x_cal)
        raw_test = model.predict_proba(x_test)
        stage_timings["hgb_calibration_and_test_inference"] = (
            time.time() - hgb_inference_started
        )
    probability_choice = select_probability_variant(
        raw_cal[calibration_fit_index],
        y_calibration_fit,
        raw_cal[calibration_selection_index],
        y_calibration_selection,
        raw_test,
    )
    probability_variants = {
        "raw": raw_test,
        "temperature": probability_choice.temperature_test,
    }
    probabilities = probability_choice.selected_test
    metric_rows.extend(
        [
            classification_metrics(
                raw_cal[calibration_selection_index],
                y_calibration_selection,
                "hgb_raw",
                "calibration_selection_unseen_org",
            ),
            classification_metrics(
                probability_choice.temperature_selection,
                y_calibration_selection,
                "hgb_temperature",
                "calibration_selection_unseen_org",
            ),
            classification_metrics(raw_test, y_test, "hgb_raw", "test_all"),
            classification_metrics(
                probability_choice.temperature_test,
                y_test,
                "hgb_temperature",
                "test_all",
            ),
            classification_metrics(
                probabilities, y_test, "hgb_selected", "test_all"
            ),
        ]
    )

    known_organizations = set(full_train["OrgId"].astype(str))
    test_known_mask = test["OrgId"].astype(str).isin(known_organizations).to_numpy()
    split_masks = {
        "test_public": test[EVALUATION_ONLY].eq("Public").to_numpy(),
        "test_private": test[EVALUATION_ONLY].eq("Private").to_numpy(),
        "test_seen_org": test_known_mask,
        "test_unseen_org": ~test_known_mask,
    }
    for name, mask in split_masks.items():
        if mask.any() and np.unique(y_test[mask]).size == 3:
            for variant, variant_probabilities in probability_variants.items():
                metric_rows.append(
                    classification_metrics(
                        variant_probabilities[mask],
                        y_test[mask],
                        f"hgb_{variant}",
                        name,
                    )
                )
            metric_rows.append(
                classification_metrics(
                    probabilities[mask], y_test[mask], "hgb_selected", name
                )
            )
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "classification_metrics.csv", index=False)

    priority = builder.priority_score(test).to_numpy(dtype=float)
    save_predictions(
        args.output_dir / "test_predictions.parquet",
        test,
        probability_variants,
        probability_choice.selected_variant,
        priority,
    )

    validation_probability_variants = {
        "raw": raw_cal[calibration_selection_index],
        "temperature": probability_choice.temperature_selection,
    }
    target_review_rates = set(float(value) for value in args.full_budget_fractions)
    target_review_rates.update(float(value) / 20.0 for value in args.ranked_budgets)
    global_thresholds = {
        variant: fit_global_tp_thresholds(values, sorted(target_review_rates))
        for variant, values in validation_probability_variants.items()
    }

    policy_started = time.time()
    print("Evaluating raw and calibrated matched-budget policies...", flush=True)
    queue_outputs: list[pd.DataFrame] = []
    aggregate_outputs: list[pd.DataFrame] = []
    stress_outputs: list[pd.DataFrame] = []
    proposed = f"joint_robust_lambda_{args.primary_lambda:g}"
    stress_policies = {
        proposed,
        "nominal_optimizer",
        "robust_no_priority",
        "priority_only",
        "priority_defer_only",
    }
    for probability_variant, variant_probabilities in probability_variants.items():
        for scope, mask in (
            ("public", test[EVALUATION_ONLY].eq("Public").to_numpy()),
            ("private", test[EVALUATION_ONLY].eq("Private").to_numpy()),
        ):
            scoped = test.loc[mask].reset_index(drop=True)
            scoped_prob = variant_probabilities[mask]
            scoped_priority = builder.priority_score(scoped).to_numpy(dtype=float)
            queue_frame, aggregate_frame, stress_frame = evaluate_policy_grid(
                scoped,
                scoped_prob,
                scoped_priority,
                scope,
                args.full_budget_fractions,
                False,
                args.priority_lambdas,
                "medium",
                args.tv_radius,
                args.automation_cost_width,
                args.close_threshold,
                args.closure_safety,
                args.seed,
                probability_variant,
                evaluate_deployment_stress=(
                    probability_variant == probability_choice.selected_variant
                ),
                deployment_stress_policies=stress_policies,
            )
            threshold_queue, threshold_aggregate, threshold_stress = (
                evaluate_global_threshold_baseline(
                    scoped,
                    scoped_prob,
                    scoped_priority,
                    scope,
                    args.full_budget_fractions,
                    False,
                    global_thresholds[probability_variant],
                    "medium",
                    args.tv_radius,
                    args.automation_cost_width,
                    args.close_threshold,
                    args.closure_safety,
                    args.seed,
                    probability_variant,
                )
            )
            queue_outputs.extend([queue_frame, threshold_queue])
            aggregate_outputs.extend([aggregate_frame, threshold_aggregate])
            if not stress_frame.empty:
                stress_outputs.append(stress_frame)
            if not threshold_stress.empty:
                stress_outputs.append(threshold_stress)

    rankings = pd.read_csv(args.rankings, dtype={"OrgId": "string", "IncidentId": "string"})
    required_rank = {"OrgId", "IncidentId", "QueueRank"}
    if not required_rank.issubset(rankings.columns):
        raise ValueError(f"Ranking file missing columns: {sorted(required_rank - set(rankings.columns))}")
    rankings["OrgId"] = rankings["OrgId"].astype("string")
    rankings["IncidentId"] = rankings["IncidentId"].astype("string")
    ranked = test.merge(rankings[list(required_rank)], on=["OrgId", "IncidentId"], how="inner", validate="one_to_one")
    if not args.allow_partial_rankings and len(ranked) != len(rankings):
        raise RuntimeError(f"Ranking merge lost rows: {len(ranked)} / {len(rankings)}")
    keyed_prob = test[["OrgId", "IncidentId"]].copy()
    for variant, variant_probabilities in probability_variants.items():
        for label_index, label in enumerate(LABELS):
            keyed_prob[f"p_{variant}_{label}"] = variant_probabilities[:, label_index]
    ranked = ranked.merge(keyed_prob, on=["OrgId", "IncidentId"], how="left", validate="one_to_one")
    ranked_priority = builder.priority_score(ranked).to_numpy(dtype=float)
    ranked_probability_variants: dict[str, np.ndarray] = {}
    for probability_variant in probability_variants:
        ranked_prob = ranked[
            [f"p_{probability_variant}_{label}" for label in LABELS]
        ].to_numpy(dtype=float)
        ranked_probability_variants[probability_variant] = ranked_prob
        ranked_queue, ranked_aggregate, ranked_stress = evaluate_policy_grid(
            ranked,
            ranked_prob,
            ranked_priority,
            "ranked",
            args.ranked_budgets,
            True,
            args.priority_lambdas,
            "medium",
            args.tv_radius,
            args.automation_cost_width,
            args.close_threshold,
            args.closure_safety,
            args.seed,
            probability_variant,
            evaluate_deployment_stress=(
                probability_variant == probability_choice.selected_variant
            ),
            deployment_stress_policies=stress_policies,
        )
        threshold_queue, threshold_aggregate, threshold_stress = (
            evaluate_global_threshold_baseline(
                ranked,
                ranked_prob,
                ranked_priority,
                "ranked",
                args.ranked_budgets,
                True,
                global_thresholds[probability_variant],
                "medium",
                args.tv_radius,
                args.automation_cost_width,
                args.close_threshold,
                args.closure_safety,
                args.seed,
                probability_variant,
            )
        )
        queue_outputs.extend([ranked_queue, threshold_queue])
        aggregate_outputs.extend([ranked_aggregate, threshold_aggregate])
        if not ranked_stress.empty:
            stress_outputs.append(ranked_stress)
        if not threshold_stress.empty:
            stress_outputs.append(threshold_stress)

    per_queue = pd.concat(queue_outputs, ignore_index=True)
    aggregate = pd.concat(aggregate_outputs, ignore_index=True)
    per_queue.to_csv(args.output_dir / "policy_metrics_by_queue.csv.gz", index=False, compression="gzip")
    aggregate.to_csv(args.output_dir / "policy_metrics_aggregate.csv", index=False)
    queue_size_strata(per_queue).to_csv(
        args.output_dir / "policy_metrics_by_queue_size.csv", index=False
    )

    print("Running pooled-capacity budget-allocation sensitivity...", flush=True)
    allocation_queue, allocation_aggregate = largest_remainder_budget_sensitivity(
        test,
        probabilities,
        builder,
        args.full_budget_fractions,
        args.primary_lambda,
        args.tv_radius,
        args.automation_cost_width,
        args.close_threshold,
        args.closure_safety,
        args.seed + 40_000,
        probability_choice.selected_variant,
    )
    allocation_queue.to_csv(
        args.output_dir / "largest_remainder_budget_sensitivity_by_queue.csv.gz",
        index=False,
        compression="gzip",
    )
    allocation_policies = {
        proposed,
        "nominal_optimizer",
        "robust_no_priority",
        "priority_defer_only",
    }
    floor_queue = per_queue[
        per_queue["scope"].isin(["public", "private"])
        & per_queue["probability_variant"].eq(probability_choice.selected_variant)
        & per_queue["policy"].isin(allocation_policies)
        & per_queue["budget_allocation_mode"].eq("per_queue_floor")
    ]
    floor_aggregate = aggregate[
        aggregate["scope"].isin(["public", "private"])
        & aggregate["probability_variant"].eq(probability_choice.selected_variant)
        & aggregate["policy"].isin(allocation_policies)
        & aggregate["budget_allocation_mode"].eq("per_queue_floor")
    ]
    pd.concat([floor_aggregate, allocation_aggregate], ignore_index=True).to_csv(
        args.output_dir / "budget_allocation_sensitivity.csv", index=False
    )
    queue_size_strata(
        pd.concat([floor_queue, allocation_queue], ignore_index=True)
    ).to_csv(
        args.output_dir / "budget_allocation_sensitivity_by_queue_size.csv",
        index=False,
    )

    deployment_stress_by_queue = pd.concat(stress_outputs, ignore_index=True)
    deployment_stress_by_queue.to_csv(
        args.output_dir / "deployment_cost_stress_by_queue.csv.gz",
        index=False,
        compression="gzip",
    )
    deployment_stress_aggregate = aggregate_deployment_stress(
        deployment_stress_by_queue
    )
    deployment_stress_aggregate.to_csv(
        args.output_dir / "deployment_cost_stress_aggregate.csv", index=False
    )
    summarize_worst_case_stress(deployment_stress_aggregate).to_csv(
        args.output_dir / "worst_case_policy_metrics.csv", index=False
    )
    stage_timings["policy_evaluation_and_cost_stress"] = time.time() - policy_started

    bootstrap_started = time.time()
    print("Running queue-clustered bootstrap comparisons...", flush=True)
    bootstrap = run_bootstraps(
        per_queue, args.primary_lambda, args.bootstrap_repetitions, args.seed + 50_000
    )
    bootstrap.to_csv(args.output_dir / "paired_queue_bootstrap.csv", index=False)
    probability_bootstrap = run_probability_ablation_bootstraps(
        per_queue,
        args.primary_lambda,
        args.bootstrap_repetitions,
        args.seed + 70_000,
    )
    probability_bootstrap.to_csv(
        args.output_dir / "probability_ablation_bootstrap.csv", index=False
    )
    stress_bootstrap = run_worst_case_stress_bootstraps(
        deployment_stress_by_queue,
        args.primary_lambda,
        probability_choice.selected_variant,
        args.bootstrap_repetitions,
        args.seed + 80_000,
    )
    stress_bootstrap.to_csv(
        args.output_dir / "worst_case_stress_bootstrap.csv", index=False
    )
    stage_timings["cluster_bootstraps"] = time.time() - bootstrap_started

    sensitivity_started = time.time()
    print("Running declared sensitivity grid on expert queues...", flush=True)
    sensitivity = sensitivity_analysis(
        ranked,
        ranked_probability_variants[probability_choice.selected_variant],
        ranked_priority,
        args.ranked_budgets,
        args.primary_lambda,
        args.seed + 90_000,
        probability_choice.selected_variant,
    )
    sensitivity.to_csv(args.output_dir / "ranked_sensitivity_analysis.csv", index=False)
    plot_results(
        aggregate,
        bootstrap,
        probability_bootstrap,
        stress_bootstrap,
        args.output_dir,
        args.primary_lambda,
        probability_choice.selected_variant,
    )
    stage_timings["sensitivity_and_figures"] = time.time() - sensitivity_started

    partition_report = {
        "full_training_incidents": int(len(full_train)),
        "excluded_unlabeled_training_incidents": unlabeled,
        "eligible_training_incidents": int(len(train)),
        "fit_incidents": int(len(fit)),
        "calibration_incidents": int(len(calibration)),
        "calibration_fit_incidents": int(len(calibration_fit)),
        "calibration_selection_incidents": int(len(calibration_selection)),
        "fit_organizations": int(fit["OrgId"].nunique()),
        "calibration_organizations": int(calibration["OrgId"].nunique()),
        "calibration_fit_organizations": int(calibration_fit["OrgId"].nunique()),
        "calibration_selection_organizations": int(
            calibration_selection["OrgId"].nunique()
        ),
        "organization_overlap_fit_calibration": int(len(set(fit["OrgId"]) & set(calibration["OrgId"]))),
        "organization_overlap_calibration_fit_selection": int(
            len(set(calibration_fit["OrgId"]) & set(calibration_selection["OrgId"]))
        ),
        "test_incidents": int(len(test)),
        "test_public_incidents": int(test[EVALUATION_ONLY].eq("Public").sum()),
        "test_private_incidents": int(test[EVALUATION_ONLY].eq("Private").sum()),
        "test_seen_organization_incidents": int(test_known_mask.sum()),
        "test_unseen_organization_incidents": int((~test_known_mask).sum()),
        "ranked_incidents_matched": int(len(ranked)),
        "ranked_organizations": int(ranked["OrgId"].nunique()),
    }
    json_dump(args.output_dir / "partition_report.json", partition_report)
    json_dump(
        args.output_dir / "model_report.json",
        {
            "temperature": probability_choice.scaler.temperature,
            "calibration_fit_log_loss_before": probability_choice.scaler.calibration_loss_before,
            "calibration_fit_log_loss_after": probability_choice.scaler.calibration_loss_after,
            "selection_log_loss_raw": probability_choice.selection_log_loss_raw,
            "selection_log_loss_temperature": probability_choice.selection_log_loss_temperature,
            "selected_probability_variant": probability_choice.selected_variant,
            "hgb_iterations_completed": int(model.n_iter_),
            "hgb_feature_names": builder.feature_names,
            "hgb_categorical_features": [
                name for name, flag in zip(builder.feature_names, builder.categorical_mask) if flag
            ],
            "linear_metadata": linear_metadata,
            "global_tp_thresholds_frozen_on_calibration_selection": global_thresholds,
        },
    )
    write_protocol_deviations(args.output_dir)
    peak_memory = peak_rss_bytes()
    manifest = {
        "pipeline_version": VERSION,
        "created_utc": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "title": STUDY_TITLE,
        "status": "empirical run; interpretation requires review of every output and declared failure criterion",
        "arguments": vars(args),
        "label_order": LABELS,
        "actions": ACTIONS,
        "cost_scenarios": {name: matrix.tolist() for name, matrix in COST_SCENARIOS.items()},
        "cost_note": "Normalized scenario units; not dollars and not stakeholder-validated monetary losses.",
        "cost_uncertainty_note": (
            "The one-sided automation-cost width applies only to close/defer costs; "
            "analyst-review cost is held fixed."
        ),
        "deployment_stress_note": (
            "Policy actions selected under the medium design matrix are frozen, then "
            "evaluated under every declared low/medium/high deployment matrix and "
            "automation-cost width in [0, 0.25, 0.50]."
        ),
        "primary_estimand_note": (
            "Worst-case normalized decision cost and regret use equal-weight "
            "organization-queue macro means. Incident-weighted results are secondary."
        ),
        "budget_allocation_note": (
            "Per-queue floor allocation is primary. Largest-remainder pooled "
            "allocation is a declared sensitivity and is not used to replace it."
        ),
        "no_probability_ablation_note": (
            "priority_defer_only uses no incident-level triage probability for "
            "review ordering or automatic actions; every unreviewed incident is deferred."
        ),
        "oracle_note": (
            "Oracle regret uses true labels only as an unattainable evaluation lower "
            "bound and never as a policy input."
        ),
        "probability_selection_note": (
            "Raw versus temperature-scaled HGB probabilities were selected by "
            "log loss on an organization-disjoint calibration-selection partition; "
            "test labels were not consulted."
        ),
        "input_sha256": {
            "train_parquet": sha256(args.train_parquet),
            "test_parquet": sha256(args.test_parquet),
            "rankings": sha256(args.rankings),
            "build_report": sha256(args.build_report),
        },
        "versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "duckdb": duckdb.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "stage_timings_seconds": {
            name: round(value, 3) for name, value in stage_timings.items()
        },
        "peak_process_rss_bytes": peak_memory,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    json_dump(args.output_dir / "run_manifest.json", manifest)
    (args.output_dir / "README_RESULTS.txt").write_text(
        "Upload guide_results_bundle.zip for manuscript analysis. Keep test_predictions.parquet locally.\n"
        "v0.6 adds the macro-queue primary estimand, a true no-probability ablation, NDCG/precision metrics, largest-remainder budget sensitivity, and confidence-interval figures.\n"
        "No result is publication evidence until the metrics, failure criteria, and sensitivity outputs are reviewed.\n",
        encoding="utf-8",
    )
    bundle = make_upload_bundle(args.output_dir)
    print(f"Experiment complete: {args.output_dir.resolve()}")
    print(f"Upload this file: {bundle.resolve()}")
    print(f"Elapsed: {manifest['elapsed_seconds']:.1f} seconds")


if __name__ == "__main__":
    main()
