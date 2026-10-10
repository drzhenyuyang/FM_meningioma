from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import RunConfig


@dataclass
class InputData:
    metadata: pd.DataFrame
    features: dict[str, dict[str, pd.DataFrame]]


def _read_feature_table(path, id_column: str, prefix: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    if id_column not in table:
        raise ValueError(f"Missing identifier column in {path.name}: {id_column}")
    if table[id_column].duplicated().any():
        raise ValueError(f"Duplicate identifiers in {path.name}")
    columns = [column for column in table.columns if column != id_column]
    if not columns:
        raise ValueError(f"No feature columns in {path.name}")
    numeric = table[columns].apply(pd.to_numeric, errors="coerce")
    numeric.columns = [f"{prefix}__{column}" for column in numeric.columns]
    return pd.concat([table[[id_column]].astype(str), numeric], axis=1)


def load_inputs(config: RunConfig) -> InputData:
    metadata = pd.read_csv(config.metadata_csv)
    endpoint_columns = {item.column for item in config.endpoints.values()}
    required = {
        config.id_column,
        config.role_column,
        config.center_column,
        *endpoint_columns,
    }
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"Metadata columns missing: {missing}")
    metadata[config.id_column] = metadata[config.id_column].astype(str)
    if metadata[config.id_column].duplicated().any():
        raise ValueError("Metadata identifiers must be unique")
    for column in endpoint_columns:
        metadata[column] = pd.to_numeric(metadata[column], errors="coerce")
        observed = set(metadata[column].dropna().unique())
        if not observed.issubset({0, 1}):
            raise ValueError(f"Endpoint {column} must contain only 0, 1, or missing")
    if config.train_role not in set(metadata[config.role_column].astype(str)):
        raise ValueError("No training rows found")

    expected = set(metadata[config.id_column])
    feature_tables: dict[str, dict[str, pd.DataFrame]] = {}
    for modality, paths in config.feature_inputs.items():
        feature_tables[modality] = {
            "brainiac": _read_feature_table(paths.brainiac, config.id_column, "brainiac"),
            "radiomics": _read_feature_table(paths.radiomics, config.id_column, "radiomics"),
        }
        for source, table in feature_tables[modality].items():
            absent = expected.difference(table[config.id_column])
            if absent:
                raise ValueError(
                    f"{source} features for {modality} are missing for {len(absent)} samples"
                )
    return InputData(metadata=metadata, features=feature_tables)


def feature_matrix(
    data: InputData,
    config: RunConfig,
    modality: str,
    method: str,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    identifier = config.id_column
    joined = data.metadata[[identifier]].copy()
    if method in {"Hybrid", "BrainIAC-only"}:
        joined = joined.merge(
            data.features[modality]["brainiac"],
            on=identifier,
            how="left",
            validate="one_to_one",
        )
    if method in {"Hybrid", "Radiomics-only"}:
        joined = joined.merge(
            data.features[modality]["radiomics"],
            on=identifier,
            how="left",
            validate="one_to_one",
        )
    if method not in {"Hybrid", "BrainIAC-only", "Radiomics-only"}:
        raise ValueError(f"Unknown method: {method}")
    columns = [column for column in joined.columns if column != identifier]
    sources = np.asarray([column.split("__", 1)[0] for column in columns], dtype=object)
    return joined[columns].to_numpy(dtype=np.float32), columns, sources
