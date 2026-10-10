from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def metric_row(frame: pd.DataFrame) -> dict:
    y = frame["y_true"].to_numpy(dtype=int)
    score = frame["discrimination_score"].to_numpy(float)
    probability = frame["calibrated_probability"].to_numpy(float)
    prediction = frame["predicted_label"].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()

    def safe(numerator, denominator):
        return numerator / denominator if denominator else np.nan

    return {
        "n": len(y),
        "positive_n": int(y.sum()),
        "prevalence": float(y.mean()),
        "auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, probability)),
        "sensitivity": safe(tp, tp + fn),
        "specificity": safe(tn, tn + fp),
        "ppv": safe(tp, tp + fp),
        "npv": safe(tn, tn + fn),
    }


def curve_rows(frame: pd.DataFrame, identity: dict) -> dict[str, list[dict]]:
    y = frame["y_true"].to_numpy(dtype=int)
    score = frame["discrimination_score"].to_numpy(float)
    probability = frame["calibrated_probability"].to_numpy(float)
    fpr, tpr, roc_threshold = roc_curve(y, score)
    precision, recall, pr_threshold = precision_recall_curve(y, score)
    observed, predicted = calibration_curve(y, probability, n_bins=10, strategy="quantile")
    dca = []
    prevalence = float(y.mean())
    for threshold in np.linspace(0.01, 0.99, 99):
        predicted_positive = probability >= threshold
        tp = np.sum(predicted_positive & (y == 1))
        fp = np.sum(predicted_positive & (y == 0))
        odds = threshold / (1.0 - threshold)
        dca.append(
            {
                **identity,
                "threshold": threshold,
                "model_net_benefit": tp / len(y) - fp / len(y) * odds,
                "treat_all_net_benefit": prevalence - (1.0 - prevalence) * odds,
                "treat_none_net_benefit": 0.0,
            }
        )
    return {
        "roc": [
            {**identity, "fpr": a, "tpr": b, "threshold": c}
            for a, b, c in zip(fpr, tpr, roc_threshold)
        ],
        "pr": [
            {
                **identity,
                "recall": recall[index],
                "precision": precision[index],
                "threshold": pr_threshold[index] if index < len(pr_threshold) else np.nan,
            }
            for index in range(len(precision))
        ],
        "calibration": [
            {**identity, "mean_predicted": a, "fraction_positive": b}
            for a, b in zip(predicted, observed)
        ],
        "dca": dca,
    }


def _bh_fdr(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def _metric_function(metric: str):
    if metric == "AUC":
        return lambda y, values: roc_auc_score(y, values), "discrimination_score", 1.0
    if metric == "AP":
        return lambda y, values: average_precision_score(y, values), "discrimination_score", 1.0
    if metric == "Brier":
        return lambda y, values: brier_score_loss(y, values), "calibrated_probability", -1.0
    raise ValueError(metric)


def paired_bootstrap(
    predictions: pd.DataFrame,
    primary_modality: str,
    single_sequence_modalities: tuple[str, ...],
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    comparisons = [
        ("Hybrid vs BrainIAC-only", "Hybrid", primary_modality, "BrainIAC-only", primary_modality),
        ("Hybrid vs Radiomics-only", "Hybrid", primary_modality, "Radiomics-only", primary_modality),
    ]
    comparisons.extend(
        (
            f"{primary_modality} vs {modality}",
            "Hybrid",
            primary_modality,
            "Hybrid",
            modality,
        )
        for modality in single_sequence_modalities
    )
    rows = []
    rng = np.random.default_rng(seed)
    labeled = predictions.dropna(subset=["y_true"])
    for evaluation in sorted(set(labeled["evaluation"]) - {"training_oof"}):
        evaluation_frame = labeled[labeled["evaluation"] == evaluation]
        for comparison_name, ref_method, ref_modality, cmp_method, cmp_modality in comparisons:
            for endpoint in sorted(evaluation_frame["endpoint"].unique()):
                reference = evaluation_frame[
                    (evaluation_frame["endpoint"] == endpoint)
                    & (evaluation_frame["method"] == ref_method)
                    & (evaluation_frame["modality"] == ref_modality)
                ].set_index("sample_id")
                comparator = evaluation_frame[
                    (evaluation_frame["endpoint"] == endpoint)
                    & (evaluation_frame["method"] == cmp_method)
                    & (evaluation_frame["modality"] == cmp_modality)
                ].set_index("sample_id")
                if reference.empty or comparator.empty:
                    continue
                if set(reference.index) != set(comparator.index):
                    raise ValueError(
                        f"Paired comparison has different samples: {evaluation}, {endpoint}, {comparison_name}"
                    )
                comparator = comparator.loc[reference.index]
                y = reference["y_true"].to_numpy(dtype=int)
                if not np.array_equal(y, comparator["y_true"].to_numpy(dtype=int)):
                    raise ValueError("Paired comparison has inconsistent labels")

                for metric in ("AUC", "AP", "Brier"):
                    function, column, direction = _metric_function(metric)
                    ref_values = reference[column].to_numpy(float)
                    cmp_values = comparator[column].to_numpy(float)
                    ref_metric = float(function(y, ref_values))
                    cmp_metric = float(function(y, cmp_values))
                    improvement = direction * (ref_metric - cmp_metric)
                    differences = []
                    for _ in range(iterations):
                        sample = rng.integers(0, len(y), len(y))
                        if np.unique(y[sample]).size < 2:
                            continue
                        differences.append(
                            direction
                            * (
                                function(y[sample], ref_values[sample])
                                - function(y[sample], cmp_values[sample])
                            )
                        )
                    differences = np.asarray(differences, dtype=float)
                    if differences.size == 0:
                        continue
                    tail = min(
                        (np.sum(differences <= 0) + 1) / (len(differences) + 1),
                        (np.sum(differences >= 0) + 1) / (len(differences) + 1),
                    )
                    rows.append(
                        {
                            "evaluation": evaluation,
                            "comparison": comparison_name,
                            "endpoint": endpoint,
                            "metric": metric,
                            "reference_value": ref_metric,
                            "comparator_value": cmp_metric,
                            "improvement_positive_favors_reference": improvement,
                            "ci_low": float(np.quantile(differences, 0.025)),
                            "ci_high": float(np.quantile(differences, 0.975)),
                            "p_value": min(1.0, 2.0 * tail),
                        }
                    )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["q_value"] = result.groupby(["evaluation", "comparison"])["p_value"].transform(
            lambda values: _bh_fdr(values.to_numpy(float))
        )
    return result
