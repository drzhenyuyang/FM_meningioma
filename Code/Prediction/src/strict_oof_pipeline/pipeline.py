from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from .calibration import PositivePlattCalibrator, youden_threshold
from .config import Candidate, RunConfig
from .data import InputData, feature_matrix, load_inputs
from .metrics import curve_rows, metric_row, paired_bootstrap
from .model import FittedBinaryMLP
from .reporting import save_curve_figures
from .selection import SelectedConfiguration, select_candidate, select_training_duration


METHODS = ("Hybrid", "BrainIAC-only", "Radiomics-only")


def _evaluation_label(role: str, train_role: str) -> str:
    return "training_oof" if role == train_role else role


def _fit_locked_model(
    data: InputData,
    config: RunConfig,
    metadata: pd.DataFrame,
    endpoint_name: str,
    endpoint_column: str,
    modality: str,
    method: str,
    selected: SelectedConfiguration,
    labeled_train: np.ndarray,
    top_k_override: dict[str, int] | None,
) -> tuple[list[dict], dict]:
    x_all, feature_names, feature_sources = feature_matrix(data, config, modality, method)
    x_train = x_all[labeled_train]
    y_all = metadata[endpoint_column].to_numpy(dtype=float)
    y_train = y_all[labeled_train].astype(int)

    calibrator = PositivePlattCalibrator().fit(selected.oof_logits, y_train)
    oof_probability = calibrator.predict(selected.oof_logits)
    threshold = youden_threshold(y_train, oof_probability)

    fitted = FittedBinaryMLP(
        selected.candidate,
        feature_names,
        feature_sources,
        device=config.device,
        top_k_override=top_k_override,
    ).fit(x_train, y_train, selected.epoch)
    logits = fitted.decision_function(x_all)
    probability = calibrator.predict(logits)

    score_origin = np.full(len(metadata), "full_training_refit", dtype=object)
    logits[labeled_train] = selected.oof_logits
    probability[labeled_train] = oof_probability
    score_origin[labeled_train] = "training_oof"

    records = []
    for index, row in metadata.iterrows():
        records.append(
            {
                "sample_id": str(row[config.id_column]),
                "endpoint": endpoint_name,
                "method": method,
                "modality": modality,
                "role": str(row[config.role_column]),
                "evaluation": _evaluation_label(
                    str(row[config.role_column]), config.train_role
                ),
                "center": str(row[config.center_column]),
                "y_true": row[endpoint_column],
                "discrimination_score": float(logits[index]),
                "calibrated_probability": float(probability[index]),
                "threshold": float(threshold),
                "predicted_label": int(probability[index] >= threshold),
                "score_origin": str(score_origin[index]),
            }
        )

    details = {
        "configuration": asdict(selected.candidate),
        "selected_epoch": int(selected.epoch),
        "training_oof_auc": float(selected.oof_auc),
        "training_oof_auc_by_epoch": selected.epoch_auc,
        "configuration_results": selected.candidate_results,
        "selected_features": fitted.selected_features,
        "calibration": {
            "type": "positive_slope_platt",
            "slope": float(calibrator.slope),
            "intercept": float(calibrator.intercept),
            "fit_source": "training_oof_logits",
        },
        "threshold": {
            "value": float(threshold),
            "criterion": "training_oof_youden_nondegenerate",
        },
    }
    return records, details


def _source_feature_counts(candidate: Candidate) -> dict[str, int]:
    if candidate.feature_route == "joint":
        return {
            "brainiac": candidate.joint_top_k,
            "radiomics": candidate.joint_top_k,
        }
    return {
        "brainiac": candidate.brainiac_top_k,
        "radiomics": candidate.radiomics_top_k,
    }


def _control_selection(
    x_train: np.ndarray,
    y_train: np.ndarray,
    candidate: Candidate,
    feature_names: list[str],
    feature_sources: np.ndarray,
    config: RunConfig,
    random_state: int,
    top_k_override: dict[str, int] | None = None,
) -> SelectedConfiguration:
    return select_training_duration(
        x_train,
        y_train,
        candidate,
        feature_names,
        feature_sources,
        config.folds,
        random_state,
        config.control_max_epochs,
        config.device,
        top_k_override=top_k_override,
    )


def _write_results(
    predictions: pd.DataFrame,
    selected_configurations: dict,
    config: RunConfig,
) -> None:
    output = config.output_dir
    predictions.to_csv(output / "predictions.csv", index=False)
    (output / "selected_configurations.json").write_text(
        json.dumps(selected_configurations, indent=2), encoding="utf-8"
    )

    labeled = predictions.dropna(subset=["y_true"]).copy()
    metric_records: list[dict] = []
    center_records: list[dict] = []
    curve_records: dict[str, list[dict]] = {
        name: [] for name in ("roc", "pr", "calibration", "dca")
    }

    grouping = ["endpoint", "method", "modality", "evaluation"]
    for identity, group in labeled.groupby(grouping, sort=False):
        if group["y_true"].nunique() < 2:
            continue
        details = dict(zip(grouping, identity))
        metric_records.append({**details, **metric_row(group)})
        curve_identity = {**details, "prevalence": float(group["y_true"].mean())}
        curves = curve_rows(group, curve_identity)
        for name in curve_records:
            curve_records[name].extend(curves[name])

    center_grouping = [*grouping, "center"]
    for identity, group in labeled.groupby(center_grouping, sort=False):
        if group["y_true"].nunique() < 2:
            continue
        details = dict(zip(center_grouping, identity))
        center_records.append({**details, **metric_row(group)})

    pd.DataFrame(metric_records).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(center_records).to_csv(output / "center_metrics.csv", index=False)
    curve_frames: dict[str, pd.DataFrame] = {}
    for name, rows in curve_records.items():
        frame = pd.DataFrame(rows)
        frame.to_csv(output / f"{name}_curves.csv", index=False)
        curve_frames[name] = frame

    comparison_frame = paired_bootstrap(
        labeled,
        config.primary_modality,
        config.single_sequence_modalities,
        config.bootstrap_iterations,
        config.base_seed,
    )
    comparison_frame.to_csv(output / "paired_comparisons.csv", index=False)
    save_curve_figures(
        curve_frames,
        output,
        config.primary_modality,
        config.single_sequence_modalities,
    )


def run(config: RunConfig) -> None:
    output = config.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    data = load_inputs(config)
    metadata = data.metadata.reset_index(drop=True)
    train_mask = metadata[config.role_column].astype(str).eq(config.train_role).to_numpy()
    predictions: list[dict] = []
    selected_configurations: dict[str, dict] = {}

    for endpoint_index, (endpoint_name, endpoint) in enumerate(config.endpoints.items()):
        y_all = metadata[endpoint.column].to_numpy(dtype=float)
        labeled_train = train_mask & np.isfinite(y_all)
        y_train = y_all[labeled_train].astype(int)
        if np.unique(y_train).size != 2:
            raise ValueError(
                f"Training labels for {endpoint_name} must contain both classes"
            )
        random_state = config.base_seed + endpoint_index * 1000
        selected_configurations[endpoint_name] = {}

        x_primary, primary_names, primary_sources = feature_matrix(
            data, config, config.primary_modality, "Hybrid"
        )
        hybrid_selected = select_candidate(
            x_primary[labeled_train],
            y_train,
            endpoint.candidates,
            primary_names,
            primary_sources,
            config.folds,
            random_state,
            config.device,
        )
        rows, details = _fit_locked_model(
            data,
            config,
            metadata,
            endpoint_name,
            endpoint.column,
            config.primary_modality,
            "Hybrid",
            hybrid_selected,
            labeled_train,
            top_k_override=None,
        )
        predictions.extend(rows)
        selected_configurations[endpoint_name][
            f"Hybrid__{config.primary_modality}"
        ] = details
        branch_counts = _source_feature_counts(hybrid_selected.candidate)

        for method, source in (
            ("BrainIAC-only", "brainiac"),
            ("Radiomics-only", "radiomics"),
        ):
            x_control, names, sources = feature_matrix(
                data, config, config.primary_modality, method
            )
            override = {source: branch_counts[source]} if source in branch_counts else None
            selected = _control_selection(
                x_control[labeled_train],
                y_train,
                hybrid_selected.candidate,
                names,
                sources,
                config,
                random_state,
                top_k_override=override,
            )
            rows, control_details = _fit_locked_model(
                data,
                config,
                metadata,
                endpoint_name,
                endpoint.column,
                config.primary_modality,
                method,
                selected,
                labeled_train,
                top_k_override=override,
            )
            predictions.extend(rows)
            selected_configurations[endpoint_name][
                f"{method}__{config.primary_modality}"
            ] = control_details

        for modality in config.single_sequence_modalities:
            x_control, names, sources = feature_matrix(data, config, modality, "Hybrid")
            selected = _control_selection(
                x_control[labeled_train],
                y_train,
                hybrid_selected.candidate,
                names,
                sources,
                config,
                random_state,
            )
            rows, control_details = _fit_locked_model(
                data,
                config,
                metadata,
                endpoint_name,
                endpoint.column,
                modality,
                "Hybrid",
                selected,
                labeled_train,
                top_k_override=None,
            )
            predictions.extend(rows)
            selected_configurations[endpoint_name][f"Hybrid__{modality}"] = (
                control_details
            )

    _write_results(pd.DataFrame(predictions), selected_configurations, config)
