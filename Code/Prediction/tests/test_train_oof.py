import numpy as np

from strict_oof_pipeline.calibration import PositivePlattCalibrator, youden_threshold
from strict_oof_pipeline.config import Candidate
from strict_oof_pipeline.selection import cross_validated_epoch_logits


def small_candidate() -> Candidate:
    return Candidate(
        feature_route="separate",
        brainiac_top_k=3,
        radiomics_top_k=3,
        joint_top_k=0,
        hidden_layers=(8,),
        activation="relu",
        dropout=0.1,
        loss="weighted_bce",
        focal_gamma=2.0,
        learning_rate=0.001,
        weight_decay=0.0,
        max_epochs=3,
        batch_size=8,
        seed=3,
    )


def test_oof_logits_and_training_calibration():
    rng = np.random.default_rng(11)
    y = np.asarray([0, 1] * 12, dtype=int)
    x = rng.normal(size=(len(y), 12)).astype(np.float32)
    x[:, 0] += y * 0.8
    names = [f"brainiac__f{i}" for i in range(6)] + [
        f"radiomics__f{i}" for i in range(6)
    ]
    sources = np.asarray(["brainiac"] * 6 + ["radiomics"] * 6, dtype=object)
    logits = cross_validated_epoch_logits(
        x,
        y,
        small_candidate(),
        names,
        sources,
        folds=3,
        random_state=17,
        device="cpu",
    )
    assert logits.shape == (3, len(y))
    assert np.isfinite(logits).all()

    calibrator = PositivePlattCalibrator().fit(logits[-1], y)
    probabilities = calibrator.predict(logits[-1])
    threshold = youden_threshold(y, probabilities)
    assert calibrator.slope > 0
    assert np.isfinite(probabilities).all()
    assert 0 <= threshold <= 1
