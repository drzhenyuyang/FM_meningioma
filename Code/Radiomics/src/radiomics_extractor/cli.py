from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from .features import (
    INTENSITY_STATISTICS,
    SEQUENCES,
    SHAPE_STATISTICS,
    extract_feature_record,
    single_sequence_feature_names,
)


def feature_group(name: str) -> str:
    if name.startswith("maskstat_"):
        return "global_mask_count"
    if name.startswith("shape_"):
        return "shape"
    if name.startswith("delta_"):
        return "inter_sequence_delta"
    return "within_sequence"


def feature_definition(name: str) -> str:
    if name.startswith("maskstat_"):
        return "voxel count"
    if name.startswith("shape_"):
        statistic = next(
            (
                item
                for item in sorted(SHAPE_STATISTICS, key=len, reverse=True)
                if name.endswith("_" + item)
            ),
            "",
        )
        return statistic.replace("_", " ")
    statistic = next(
        (item for item in sorted(INTENSITY_STATISTICS, key=len, reverse=True) if name.endswith("_" + item)),
        "",
    )
    return "voxel-support count" if statistic == "nvox" else statistic


def extract(manifest_path: Path, output_path: Path) -> None:
    manifest = pd.read_csv(manifest_path)
    required = {"sample_id", "mask_path", *(f"{sequence}_path" for sequence in SEQUENCES)}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest columns missing: {sorted(missing)}")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Sample identifiers must be unique")

    rows = []
    errors = []
    manifest_directory = manifest_path.resolve().parent

    def resolve_input(value) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else manifest_directory / path

    for row in manifest.itertuples(index=False):
        try:
            mask_image = nib.load(str(resolve_input(row.mask_path)))
            mask = np.asanyarray(mask_image.dataobj).astype(np.int16)
            images = {}
            for sequence in SEQUENCES:
                image = nib.load(str(resolve_input(getattr(row, f"{sequence}_path"))))
                if image.shape != mask_image.shape or not np.allclose(
                    image.affine, mask_image.affine, atol=1e-4
                ):
                    raise ValueError(f"Geometry mismatch for {sequence}")
                images[sequence] = image.get_fdata(dtype=np.float32)
            rows.append({"sample_id": str(row.sample_id), **extract_feature_record(mask, images)})
        except Exception as error:
            errors.append({"sample_id": str(row.sample_id), "error": repr(error)})

    if not rows:
        raise RuntimeError("No samples were extracted successfully")
    multimodal = pd.DataFrame(rows)
    if multimodal.shape[1] != 448:
        raise RuntimeError("The multimodal output must contain sample_id plus 447 features")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    multimodal.to_csv(output_path, index=False)
    feature_names = [column for column in multimodal.columns if column != "sample_id"]
    for sequence in SEQUENCES:
        names = single_sequence_feature_names(feature_names, sequence)
        multimodal[["sample_id", *names]].to_csv(
            output_path.with_name(f"radiomics_{sequence}_90.csv"), index=False
        )
    pd.DataFrame(
        {
            "feature_name": feature_names,
            "feature_group": [feature_group(name) for name in feature_names],
            "definition": [feature_definition(name) for name in feature_names],
        }
    ).to_csv(output_path.with_name("radiomics_feature_dictionary.csv"), index=False)
    if errors:
        pd.DataFrame(errors).to_csv(output_path.with_name("radiomics_errors.csv"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract custom 90/447 MRI descriptors")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    extract(arguments.manifest, arguments.output)


if __name__ == "__main__":
    main()
