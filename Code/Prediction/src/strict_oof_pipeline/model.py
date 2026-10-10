from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import ttest_ind
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import Candidate


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def univariate_association_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    negative = x[y == 0]
    positive = x[y == 1]
    statistic, _ = ttest_ind(
        negative,
        positive,
        axis=0,
        equal_var=False,
        nan_policy="omit",
    )
    return np.nan_to_num(
        np.abs(statistic),
        nan=0.0,
        posinf=np.finfo(np.float32).max,
        neginf=0.0,
    )


def correlation_prune(
    x: np.ndarray,
    ranked_indices: np.ndarray,
    top_k: int,
    threshold: float = 0.985,
) -> list[int]:
    selected: list[int] = []
    for raw_index in ranked_indices:
        index = int(raw_index)
        keep = True
        for previous in selected:
            correlation = np.corrcoef(x[:, index], x[:, previous])[0, 1]
            if np.isfinite(correlation) and abs(float(correlation)) >= threshold:
                keep = False
                break
        if keep:
            selected.append(index)
        if len(selected) >= min(top_k, x.shape[1]):
            break
    return selected


@dataclass
class PreparedSource:
    source: str
    original_indices: np.ndarray
    feature_names: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler
    standardized: np.ndarray


@dataclass
class FittedFeatureBlock:
    source: str
    original_indices: np.ndarray
    selected_local_indices: list[int]
    selected_feature_names: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler

    def transform(self, x: np.ndarray) -> np.ndarray:
        standardized = self.scaler.transform(
            self.imputer.transform(x[:, self.original_indices])
        )
        return standardized[:, self.selected_local_indices].astype(np.float32)


def _prepare_source(
    x: np.ndarray,
    feature_names: list[str],
    feature_sources: np.ndarray,
    source: str,
) -> PreparedSource:
    source_indices = np.flatnonzero(feature_sources == source)
    if len(source_indices) == 0:
        raise ValueError(f"No {source} features are available")

    initial = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(
        x[:, source_indices]
    )
    variable = np.nanvar(initial, axis=0) > 1e-10
    original_indices = source_indices[variable]
    names = [feature_names[index] for index in original_indices]
    if len(original_indices) == 0:
        raise ValueError(f"No variable {source} features remain")

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(
        imputer.fit_transform(x[:, original_indices])
    ).astype(np.float32)
    return PreparedSource(
        source=source,
        original_indices=original_indices,
        feature_names=names,
        imputer=imputer,
        scaler=scaler,
        standardized=standardized,
    )


def fit_feature_blocks(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    feature_sources: np.ndarray,
    candidate: Candidate,
    top_k_override: dict[str, int] | None = None,
) -> list[FittedFeatureBlock]:
    present_sources = [
        source for source in ("brainiac", "radiomics") if source in feature_sources
    ]
    prepared = {
        source: _prepare_source(x, feature_names, feature_sources, source)
        for source in present_sources
    }

    selected: dict[str, list[int]] = {source: [] for source in present_sources}
    if candidate.feature_route == "joint" and len(present_sources) == 2:
        matrices = [prepared[source].standardized for source in present_sources]
        concatenated = np.concatenate(matrices, axis=1)
        mapping = [
            (source, local_index)
            for source in present_sources
            for local_index in range(prepared[source].standardized.shape[1])
        ]
        ranking = np.argsort(
            -univariate_association_scores(concatenated, y),
            kind="stable",
        )
        joint_indices = correlation_prune(
            concatenated,
            ranking,
            candidate.joint_top_k,
        )
        for joint_index in joint_indices:
            source, local_index = mapping[joint_index]
            selected[source].append(local_index)
    else:
        for source in present_sources:
            block = prepared[source]
            default_k = (
                candidate.brainiac_top_k
                if source == "brainiac"
                else candidate.radiomics_top_k
            )
            top_k = int((top_k_override or {}).get(source, default_k))
            ranking = np.argsort(
                -univariate_association_scores(block.standardized, y),
                kind="stable",
            )
            selected[source] = correlation_prune(
                block.standardized,
                ranking,
                top_k,
            )

    blocks = []
    for source in present_sources:
        block = prepared[source]
        local_indices = selected[source]
        if not local_indices:
            continue
        blocks.append(
            FittedFeatureBlock(
                source=source,
                original_indices=block.original_indices,
                selected_local_indices=local_indices,
                selected_feature_names=[
                    block.feature_names[index] for index in local_indices
                ],
                imputer=block.imputer,
                scaler=block.scaler,
            )
        )
    if not blocks:
        raise ValueError("No features were selected")
    return blocks


class MLP(nn.Module):
    def __init__(self, input_dim: int, candidate: Candidate):
        super().__init__()
        activation = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }.get(candidate.activation)
        if activation is None:
            raise ValueError(f"Unsupported activation: {candidate.activation}")
        layers: list[nn.Module] = []
        previous = input_dim
        for width in candidate.hidden_layers:
            layers.extend(
                [
                    nn.Linear(previous, width),
                    activation(),
                    nn.Dropout(candidate.dropout),
                ]
            )
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def _loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    candidate: Candidate,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    weighted_bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    if candidate.loss == "weighted_bce":
        return weighted_bce.mean()
    probability = torch.sigmoid(logits)
    p_t = targets * probability + (1.0 - targets) * (1.0 - probability)
    return (((1.0 - p_t) ** candidate.focal_gamma) * weighted_bce).mean()


class FittedBinaryMLP:
    def __init__(
        self,
        candidate: Candidate,
        feature_names: list[str],
        feature_sources: np.ndarray,
        device: str | None = None,
        top_k_override: dict[str, int] | None = None,
    ):
        self.candidate = candidate
        self.feature_names = feature_names
        self.feature_sources = np.asarray(feature_sources)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.top_k_override = top_k_override
        self.blocks: list[FittedFeatureBlock] = []
        self.model: MLP | None = None

    @property
    def selected_features(self) -> dict[str, list[str]]:
        return {block.source: block.selected_feature_names for block in self.blocks}

    def _transform(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate([block.transform(x) for block in self.blocks], axis=1)

    def _prepare_training(self, x: np.ndarray, y: np.ndarray, seed: int):
        _set_seed(seed)
        self.blocks = fit_feature_blocks(
            x,
            y,
            self.feature_names,
            self.feature_sources,
            self.candidate,
            self.top_k_override,
        )
        transformed = self._transform(x)
        self.model = MLP(transformed.shape[1], self.candidate).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.candidate.learning_rate,
            weight_decay=self.candidate.weight_decay,
        )
        positives = max(int(y.sum()), 1)
        negatives = max(int(len(y) - y.sum()), 1)
        pos_weight = torch.tensor(
            negatives / positives,
            dtype=torch.float32,
            device=self.device,
        )
        dataset = TensorDataset(
            torch.as_tensor(transformed, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
        )
        loader = DataLoader(
            dataset,
            batch_size=min(self.candidate.batch_size, len(dataset)),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        return optimizer, pos_weight, loader

    def _train_one_epoch(self, optimizer, pos_weight, loader) -> None:
        if self.model is None:
            raise RuntimeError("Model has not been initialized")
        self.model.train()
        for features, targets in loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(self.model(features), targets, self.candidate, pos_weight)
            loss.backward()
            optimizer.step()

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int,
        seed_offset: int = 0,
    ) -> "FittedBinaryMLP":
        seed = self.candidate.seed + seed_offset
        optimizer, pos_weight, loader = self._prepare_training(x, y, seed)
        for _ in range(epochs):
            self._train_one_epoch(optimizer, pos_weight, loader)
        return self

    def validation_curve(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_validation: np.ndarray,
        max_epochs: int,
        seed_offset: int = 0,
    ) -> np.ndarray:
        seed = self.candidate.seed + seed_offset
        optimizer, pos_weight, loader = self._prepare_training(x, y, seed)
        predictions = []
        for _ in range(max_epochs):
            self._train_one_epoch(optimizer, pos_weight, loader)
            predictions.append(self.decision_function(x_validation))
        return np.asarray(predictions)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted")
        transformed = self._transform(x)
        self.model.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(
                transformed,
                dtype=torch.float32,
                device=self.device,
            )
            return self.model(tensor).cpu().numpy().astype(float)

