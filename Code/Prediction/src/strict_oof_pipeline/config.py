from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FeatureInput:
    brainiac: Path
    radiomics: Path


@dataclass(frozen=True)
class Candidate:
    feature_route: str
    brainiac_top_k: int
    radiomics_top_k: int
    joint_top_k: int
    hidden_layers: tuple[int, ...]
    activation: str
    dropout: float
    loss: str
    focal_gamma: float
    learning_rate: float
    weight_decay: float
    max_epochs: int
    batch_size: int
    seed: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Candidate":
        route = str(value.get("feature_route", "separate")).lower()
        if route not in {"separate", "joint"}:
            raise ValueError(f"Unsupported feature route: {route}")
        loss = str(value.get("loss", "weighted_bce")).lower()
        if loss not in {"weighted_bce", "focal"}:
            raise ValueError(f"Unsupported loss: {loss}")
        candidate = cls(
            feature_route=route,
            brainiac_top_k=int(value.get("brainiac_top_k", 32)),
            radiomics_top_k=int(value.get("radiomics_top_k", 32)),
            joint_top_k=int(value.get("joint_top_k", 0)),
            hidden_layers=tuple(int(x) for x in value["hidden_layers"]),
            activation=str(value["activation"]).lower(),
            dropout=float(value["dropout"]),
            loss=loss,
            focal_gamma=float(value.get("focal_gamma", 2.0)),
            learning_rate=float(value["learning_rate"]),
            weight_decay=float(value["weight_decay"]),
            max_epochs=int(value.get("max_epochs", 200)),
            batch_size=int(value["batch_size"]),
            seed=int(value["seed"]),
        )
        if candidate.feature_route == "joint" and candidate.joint_top_k < 2:
            raise ValueError("joint_top_k must be at least 2 for joint selection")
        if candidate.max_epochs < 1 or candidate.batch_size < 1:
            raise ValueError("max_epochs and batch_size must be positive")
        if not candidate.hidden_layers or any(width < 1 for width in candidate.hidden_layers):
            raise ValueError("hidden_layers must contain positive widths")
        return candidate


@dataclass(frozen=True)
class EndpointConfig:
    column: str
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class RunConfig:
    metadata_csv: Path
    output_dir: Path
    id_column: str
    role_column: str
    center_column: str
    train_role: str
    feature_inputs: dict[str, FeatureInput]
    primary_modality: str
    single_sequence_modalities: tuple[str, ...]
    endpoints: dict[str, EndpointConfig]
    folds: int
    control_max_epochs: int
    bootstrap_iterations: int
    base_seed: int
    device: str | None


def load_config(path: str | Path) -> RunConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    base = config_path.parent

    def resolve(value: str) -> Path:
        item = Path(value)
        return item if item.is_absolute() else (base / item).resolve()

    inputs = {
        str(modality): FeatureInput(
            brainiac=resolve(paths["brainiac"]),
            radiomics=resolve(paths["radiomics"]),
        )
        for modality, paths in raw["feature_inputs"].items()
    }
    endpoints = {
        str(name): EndpointConfig(
            column=str(spec.get("column", name)),
            candidates=tuple(Candidate.from_dict(item) for item in spec["candidates"]),
        )
        for name, spec in raw["endpoints"].items()
    }
    if any(not item.candidates for item in endpoints.values()):
        raise ValueError("Every endpoint must define at least one configuration")

    primary = str(raw.get("primary_modality", "Multimodal"))
    single = tuple(str(x) for x in raw.get("single_sequence_modalities", []))
    missing_modalities = [item for item in (primary, *single) if item not in inputs]
    if missing_modalities:
        raise ValueError(f"Feature inputs missing for modalities: {missing_modalities}")

    return RunConfig(
        metadata_csv=resolve(raw["metadata_csv"]),
        output_dir=resolve(raw["output_dir"]),
        id_column=str(raw.get("id_column", "sample_id")),
        role_column=str(raw.get("role_column", "role")),
        center_column=str(raw.get("center_column", "center")),
        train_role=str(raw.get("train_role", "train")),
        feature_inputs=inputs,
        primary_modality=primary,
        single_sequence_modalities=single,
        endpoints=endpoints,
        folds=int(raw.get("folds", 5)),
        control_max_epochs=int(raw.get("control_max_epochs", 200)),
        bootstrap_iterations=int(raw.get("bootstrap_iterations", 1000)),
        base_seed=int(raw.get("base_seed", 2026)),
        device=str(raw["device"]) if raw.get("device") else None,
    )

