import numpy as np

from radiomics_extractor.features import (
    DELTA_PAIRS,
    SEQUENCES,
    extract_feature_record,
    single_sequence_feature_names,
)


def synthetic_case():
    rng = np.random.default_rng(7)
    shape = (16, 18, 12)
    mask = np.zeros(shape, dtype=np.int16)
    mask[3:9, 4:11, 3:8] = 1
    mask[9:12, 6:13, 4:9] = 2
    mask[5:8, 8:12, 5:9] = 3
    images = {name: rng.normal(size=shape).astype(np.float32) for name in SEQUENCES}
    return mask, images


def test_feature_counts_and_delta_order():
    mask, images = synthetic_case()
    record = extract_feature_record(mask, images)
    names = list(record)
    assert len(names) == 447
    assert DELTA_PAIRS == (
        ("T1ce", "T1"),
        ("T2", "T1"),
        ("FLAIR", "T2"),
        ("T1ce", "FLAIR"),
    )
    for sequence in SEQUENCES:
        assert len(single_sequence_feature_names(names, sequence)) == 90


def test_feature_family_partition():
    mask, images = synthetic_case()
    names = list(extract_feature_record(mask, images))
    assert sum(name.startswith("maskstat_") for name in names) == 6
    assert sum(name.startswith("shape_") for name in names) == 33
    assert sum(name.startswith("delta_") for name in names) == 204
    assert len(names) - 6 - 33 - 204 == 204
