from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import log_loss, roc_curve


class PositivePlattCalibrator:
    def __init__(self):
        self.slope = 1.0
        self.intercept = 0.0

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "PositivePlattCalibrator":
        logits = np.asarray(logits, dtype=float)
        y = np.asarray(y, dtype=int)
        def objective(parameters):
            slope = np.exp(parameters[0])
            probabilities = expit(slope * logits + parameters[1])
            return log_loss(y, probabilities, labels=[0, 1])

        solution = minimize(objective, np.array([0.0, 0.0]), method="BFGS")
        if not solution.success and not np.isfinite(solution.fun):
            raise RuntimeError("Positive-slope Platt calibration did not converge")
        self.slope = float(np.exp(solution.x[0]))
        self.intercept = float(solution.x[1])
        return self

    def predict(self, logits: np.ndarray) -> np.ndarray:
        return expit(self.slope * np.asarray(logits) + self.intercept)


def youden_threshold(y: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y, probabilities)
    candidates = []
    for fpr, tpr, threshold in zip(false_positive_rate, true_positive_rate, thresholds):
        if not np.isfinite(threshold):
            continue
        prediction = probabilities >= threshold
        if prediction.all() or not prediction.any():
            continue
        candidates.append(
            (
                float(tpr - fpr),
                -float(np.hypot(fpr, 1.0 - tpr)),
                float(threshold),
            )
        )
    if not candidates:
        finite = np.isfinite(thresholds)
        distances = np.hypot(false_positive_rate[finite], 1.0 - true_positive_rate[finite])
        return float(thresholds[finite][int(np.argmin(distances))])
    return max(candidates)[2]
