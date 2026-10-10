from __future__ import annotations

import numpy as np


SEQUENCES = ("T1", "T1ce", "T2", "FLAIR")
REGIONS = ("tumor13", "edema2", "whole123")
DELTA_PAIRS = (("T1ce", "T1"), ("T2", "T1"), ("FLAIR", "T2"), ("T1ce", "FLAIR"))
INTENSITY_STATISTICS = (
    "nvox", "mean", "std", "min", "p05", "p10", "p25", "p50", "p75",
    "p90", "p95", "max", "iqr", "skew", "kurtosis", "energy", "entropy",
)
SHAPE_STATISTICS = (
    "voxels", "log_voxels", "extent_x", "extent_y", "extent_z", "bbox_volume",
    "fill_ratio", "centroid_x", "centroid_y", "centroid_z", "max_axial_area",
)


def robust_z(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    nonzero = finite[finite != 0]
    values = nonzero if nonzero.size > 100 else finite
    if values.size == 0:
        return image.astype(np.float32)
    median = np.nanmedian(values)
    first, third = np.nanpercentile(values, [25, 75])
    scale = third - first
    if not np.isfinite(scale) or scale < 1e-6:
        scale = np.nanstd(values)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return np.clip((image - median) / scale, -10, 10).astype(np.float32)


def intensity_features(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values)
    result: dict[str, float] = {f"{prefix}_nvox": int(values.size)}
    values = values[np.isfinite(values)]
    if values.size < 5:
        result.update({f"{prefix}_{name}": np.nan for name in INTENSITY_STATISTICS[1:]})
        return result
    percentiles = np.percentile(values, [5, 10, 25, 50, 75, 90, 95])
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values))
    standardized = (values - mean) / (standard_deviation + 1e-6)
    histogram, _ = np.histogram(values, bins=32, range=(-6, 6))
    probability = histogram.astype(float) / max(1, histogram.sum())
    probability = probability[probability > 0]
    result.update(
        {
            f"{prefix}_mean": mean,
            f"{prefix}_std": standard_deviation,
            f"{prefix}_min": float(np.min(values)),
            f"{prefix}_p05": float(percentiles[0]),
            f"{prefix}_p10": float(percentiles[1]),
            f"{prefix}_p25": float(percentiles[2]),
            f"{prefix}_p50": float(percentiles[3]),
            f"{prefix}_p75": float(percentiles[4]),
            f"{prefix}_p90": float(percentiles[5]),
            f"{prefix}_p95": float(percentiles[6]),
            f"{prefix}_max": float(np.max(values)),
            f"{prefix}_iqr": float(percentiles[4] - percentiles[2]),
            f"{prefix}_skew": float(np.mean(standardized**3)),
            f"{prefix}_kurtosis": float(np.mean(standardized**4) - 3.0),
            f"{prefix}_energy": float(np.mean(values * values)),
            f"{prefix}_entropy": float(-(probability * np.log2(probability)).sum()),
        }
    )
    return result


def shape_features(mask: np.ndarray, prefix: str) -> dict[str, float]:
    voxel_count = int(mask.sum())
    result: dict[str, float] = {
        f"{prefix}_voxels": voxel_count,
        f"{prefix}_log_voxels": float(np.log1p(voxel_count)),
    }
    if voxel_count == 0:
        result.update(
            {
                f"{prefix}_{name}": np.nan if name.startswith("centroid") else 0.0
                for name in SHAPE_STATISTICS[2:]
            }
        )
        return result
    coordinates = np.argwhere(mask)
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    extent = maximum - minimum + 1
    bounding_box_volume = int(np.prod(extent))
    centroid = coordinates.mean(axis=0) / np.asarray(mask.shape)
    axial_counts = np.bincount(coordinates[:, 2], minlength=mask.shape[2])
    result.update(
        {
            f"{prefix}_extent_x": int(extent[0]),
            f"{prefix}_extent_y": int(extent[1]),
            f"{prefix}_extent_z": int(extent[2]),
            f"{prefix}_bbox_volume": bounding_box_volume,
            f"{prefix}_fill_ratio": float(voxel_count / max(1, bounding_box_volume)),
            f"{prefix}_centroid_x": float(centroid[0]),
            f"{prefix}_centroid_y": float(centroid[1]),
            f"{prefix}_centroid_z": float(centroid[2]),
            f"{prefix}_max_axial_area": int(axial_counts.max()),
        }
    )
    return result


def extract_feature_record(mask: np.ndarray, images: dict[str, np.ndarray]) -> dict[str, float]:
    if set(images) != set(SEQUENCES):
        raise ValueError(f"Images must contain exactly: {SEQUENCES}")
    if any(image.shape != mask.shape for image in images.values()):
        raise ValueError("Image and mask arrays must share shape")
    regions = {
        "tumor13": (mask == 1) | (mask == 3),
        "edema2": mask == 2,
        "whole123": (mask >= 1) & (mask <= 3),
    }
    result: dict[str, float] = {
        "maskstat_label1_voxels": int(np.sum(mask == 1)),
        "maskstat_label2_voxels": int(np.sum(mask == 2)),
        "maskstat_label3_voxels": int(np.sum(mask == 3)),
        "maskstat_tumor13_voxels": int(regions["tumor13"].sum()),
        "maskstat_edema2_voxels": int(regions["edema2"].sum()),
        "maskstat_whole123_voxels": int(regions["whole123"].sum()),
    }
    for region_name, region_mask in regions.items():
        result.update(shape_features(region_mask, f"shape_{region_name}"))
    normalized = {name: robust_z(image) for name, image in images.items()}
    for sequence in SEQUENCES:
        for region_name, region_mask in regions.items():
            result.update(
                intensity_features(
                    normalized[sequence][region_mask], f"{sequence}_{region_name}"
                )
            )
    for first, second in DELTA_PAIRS:
        difference = normalized[first] - normalized[second]
        for region_name, region_mask in regions.items():
            result.update(
                intensity_features(
                    difference[region_mask],
                    f"delta_{first}_minus_{second}_{region_name}",
                )
            )
    if len(result) != 447:
        raise RuntimeError(f"Expected 447 features, generated {len(result)}")
    return result


def single_sequence_feature_names(all_names: list[str], sequence: str) -> list[str]:
    shared = [name for name in all_names if name.startswith(("maskstat_", "shape_"))]
    sequence_specific = [name for name in all_names if name.startswith(f"{sequence}_")]
    names = shared + sequence_specific
    if len(names) != 90:
        raise RuntimeError(f"Expected 90 {sequence} features, generated {len(names)}")
    return names
