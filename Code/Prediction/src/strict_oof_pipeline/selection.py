from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import Candidate
from .model import FittedBinaryMLP


@dataclass
class SelectedConfiguration:
    candidate: Candidate
    epoch: int
    oof_auc: float
    oof_logits: np.ndarray
    epoch_auc: list[float]
    candidate_results: list[dict]


def _fold_count(y: np.ndarray, requested: int) -> int:
    counts = np.bincount(y.astype(int), minlength=2)
    folds = min(requested, int(counts.min()))
    if folds < 2:
        raise ValueError("At least two samples from each class are required")
    return folds


def fold_indices(y: np.ndarray, folds: int, random_state: int):
    splitter = StratifiedKFold(
        n_splits=_fold_count(y, folds),
        shuffle=True,
        random_state=random_state,
    )
    return list(splitter.split(np.zeros(len(y)), y))


def cross_validated_epoch_logits(
    x: np.ndarray,
    y: np.ndarray,
    candidate: Candidate,
    feature_names: list[str],
    feature_sources: np.ndarray,
    folds: int,
    random_state: int,
    device: str | None,
    max_epochs: int | None = None,
    top_k_override: dict[str, int] | None = None,
) -> np.ndarray:
    epochs = int(max_epochs or candidate.max_epochs)
    result = np.full((epochs, len(y)), np.nan, dtype=float)
    for fold, (train_index, valid_index) in enumerate(fold_indices(y, folds, random_state)):
        model = FittedBinaryMLP(
            candidate,
            feature_names,
            feature_sources,
            device=device,
            top_k_override=top_k_override,
        )
        result[:, valid_index] = model.validation_curve(
            x[train_index],
            y[train_index],
            x[valid_index],
            epochs,
            seed_offset=fold * 1000,
        )
    return result


def epoch_auc_curve(y: np.ndarray, logits: np.ndarray) -> list[float]:
    return [float(roc_auc_score(y, logits[index])) for index in range(logits.shape[0])]


def select_candidate(
    x: np.ndarray,
    y: np.ndarray,
    candidates: tuple[Candidate, ...],
    feature_names: list[str],
    feature_sources: np.ndarray,
    folds: int,
    random_state: int,
    device: str | None,
) -> SelectedConfiguration:
    records = []
    cached_logits = []
    cached_curves = []
    for index, candidate in enumerate(candidates):
        logits = cross_validated_epoch_logits(
            x,
            y,
            candidate,
            feature_names,
            feature_sources,
            folds,
            random_state,
            device,
        )
        curve = epoch_auc_curve(y, logits)
        best_epoch = int(np.nanargmax(curve)) + 1
        records.append(
            {
                "configuration": index + 1,
                "best_epoch": best_epoch,
                "training_oof_auc": curve[best_epoch - 1],
                **asdict(candidate),
            }
        )
        cached_logits.append(logits)
        cached_curves.append(curve)
    best_index = max(
        range(len(records)),
        key=lambda index: (records[index]["training_oof_auc"], -index),
    )
    epoch = int(records[best_index]["best_epoch"])
    return SelectedConfiguration(
        candidate=candidates[best_index],
        epoch=epoch,
        oof_auc=float(records[best_index]["training_oof_auc"]),
        oof_logits=cached_logits[best_index][epoch - 1].copy(),
        epoch_auc=cached_curves[best_index],
        candidate_results=records,
    )


def select_training_duration(
    x: np.ndarray,
    y: np.ndarray,
    candidate: Candidate,
    feature_names: list[str],
    feature_sources: np.ndarray,
    folds: int,
    random_state: int,
    max_epochs: int,
    device: str | None,
    top_k_override: dict[str, int] | None = None,
) -> SelectedConfiguration:
    logits = cross_validated_epoch_logits(
        x,
        y,
        candidate,
        feature_names,
        feature_sources,
        folds,
        random_state,
        device,
        max_epochs=max_epochs,
        top_k_override=top_k_override,
    )
    curve = epoch_auc_curve(y, logits)
    epoch = int(np.nanargmax(curve)) + 1
    record = {
        "configuration": 1,
        "best_epoch": epoch,
        "training_oof_auc": curve[epoch - 1],
        **asdict(candidate),
    }
    return SelectedConfiguration(
        candidate=candidate,
        epoch=epoch,
        oof_auc=float(curve[epoch - 1]),
        oof_logits=logits[epoch - 1].copy(),
        epoch_auc=curve,
        candidate_results=[record],
    )
