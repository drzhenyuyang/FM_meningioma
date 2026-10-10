from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return cleaned or "result"


def _draw_curve(
    frame: pd.DataFrame,
    curve_name: str,
    line_column: str,
    title: str,
    destination: Path,
) -> None:
    specifications = {
        "roc": ("fpr", "tpr", "False-positive rate", "True-positive rate"),
        "pr": ("recall", "precision", "Recall", "Precision"),
        "calibration": (
            "mean_predicted",
            "fraction_positive",
            "Predicted probability",
            "Observed frequency",
        ),
        "dca": (
            "threshold",
            "model_net_benefit",
            "Threshold probability",
            "Net benefit",
        ),
    }
    x, y, xlabel, ylabel = specifications[curve_name]
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=180)
    for label, line in frame.groupby(line_column, sort=False):
        line = line.sort_values(x)
        ax.plot(line[x], line[y], linewidth=1.8, label=str(label))

    if curve_name == "roc":
        ax.plot([0, 1], [0, 1], "--", color="0.55", linewidth=1, label="Chance")
    elif curve_name == "pr":
        prevalence = float(frame["prevalence"].iloc[0]) if "prevalence" in frame else None
        if prevalence is not None and np.isfinite(prevalence):
            ax.axhline(prevalence, linestyle="--", color="0.55", linewidth=1, label="Prevalence")
    elif curve_name == "calibration":
        ax.plot([0, 1], [0, 1], "--", color="0.55", linewidth=1, label="Ideal")
    elif curve_name == "dca":
        baseline = frame.sort_values("threshold").drop_duplicates("threshold")
        ax.plot(
            baseline["threshold"],
            baseline["treat_all_net_benefit"],
            "--",
            color="0.45",
            linewidth=1.2,
            label="Treat all",
        )
        ax.axhline(0, linestyle=":", color="black", linewidth=1.2, label="Treat none")
        finite = frame["model_net_benefit"].replace([np.inf, -np.inf], np.nan).dropna()
        if not finite.empty:
            lower = min(-0.02, float(finite.quantile(0.01)) - 0.02)
            upper = max(0.05, float(finite.quantile(0.99)) + 0.02)
            ax.set_ylim(lower, upper)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def save_curve_figures(
    curves: dict[str, pd.DataFrame],
    output_dir: Path,
    primary_modality: str,
    single_sequence_modalities: tuple[str, ...],
) -> None:
    figure_dir = output_dir / "figures"
    method_dir = figure_dir / "method_comparison"
    modality_dir = figure_dir / "modality_comparison"
    method_dir.mkdir(parents=True, exist_ok=True)
    modality_dir.mkdir(parents=True, exist_ok=True)

    for curve_name, frame in curves.items():
        if frame.empty:
            continue
        for (endpoint, evaluation), group in frame.groupby(
            ["endpoint", "evaluation"], sort=False
        ):
            method_frame = group[group["modality"] == primary_modality].copy()
            if not method_frame.empty:
                _draw_curve(
                    method_frame,
                    curve_name,
                    "method",
                    f"{endpoint}: method comparison ({evaluation})",
                    method_dir
                    / f"{_slug(endpoint)}__{_slug(evaluation)}__{curve_name}.png",
                )

            allowed = {primary_modality, *single_sequence_modalities}
            modality_frame = group[
                (group["method"] == "Hybrid") & group["modality"].isin(allowed)
            ].copy()
            if not modality_frame.empty:
                _draw_curve(
                    modality_frame,
                    curve_name,
                    "modality",
                    f"{endpoint}: sequence comparison ({evaluation})",
                    modality_dir
                    / f"{_slug(endpoint)}__{_slug(evaluation)}__{curve_name}.png",
                )
