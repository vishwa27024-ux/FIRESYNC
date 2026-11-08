
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

ARCHIVE_PATH = ROOT / "data/processed/firms_archive_normalized_2021_2025.parquet"
MARCH_PATH = ROOT / "data/processed/march_2025_normalized_observations.parquet"

OUTPUT_DIR = ROOT / "data/interim"
DETAIL_PATH = OUTPUT_DIR / "archive_march_attribute_comparison.csv"
REPORT_PATH = OUTPUT_DIR / "archive_march_attribute_report.json"

KEYS = [
    "sensor_family",
    "satellite_platform",
    "acq_datetime_utc",
    "latitude_key",
    "longitude_key",
]

NUMERIC_FIELDS = [
    "confidence_numeric",
    "modis_brightness_21_22_k",
    "modis_brightness_t31_k",
    "viirs_brightness_i4_k",
    "viirs_brightness_i5_k",
    "frp_mw",
    "scan_native",
    "track_native",
]

TEXT_FIELDS = [
    "confidence_native",
    "confidence_scheme",
    "confidence_class",
    "daynight",
    "version_native",
    "hotspot_type_native",
    "instrument_native",
]

PRODUCT_MAP = {
    "MODIS_C61_ARCHIVE": "MODIS_SP",
    "VIIRS_NOAA20_C2_ARCHIVE": "VIIRS_NOAA20_SP",
    "VIIRS_SNPP_C2_ARCHIVE": "VIIRS_SNPP_SP",
    "VIIRS_NOAA21_C2_ARCHIVE": "VIIRS_NOAA21_SP",
}

ATOL = 1e-6


def prepare(frame, name):
    frame = frame.copy()

    frame["acq_datetime_utc"] = pd.to_datetime(
        frame["acq_datetime_utc"], utc=True, errors="coerce"
    )

    if frame["acq_datetime_utc"].isna().any():
        raise ValueError(f"{name}: invalid UTC timestamps")

    frame["latitude_key"] = pd.to_numeric(
        frame["latitude"], errors="coerce"
    ).round(6)

    frame["longitude_key"] = pd.to_numeric(
        frame["longitude"], errors="coerce"
    ).round(6)

    if frame[["latitude_key", "longitude_key"]].isna().any().any():
        raise ValueError(f"{name}: invalid coordinates")

    return frame


def compare_numeric(left, right):
    left = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)

    return np.isclose(
        left, right, rtol=0, atol=ATOL, equal_nan=True
    )


def compare_text(left, right):
    left = left.astype("string").fillna("<NULL>").str.strip()
    right = right.astype("string").fillna("<NULL>").str.strip()

    return left.eq(right).to_numpy()


def main():
    if not ARCHIVE_PATH.exists() or not MARCH_PATH.exists():
        raise FileNotFoundError("One or both input datasets are missing.")

    if DETAIL_PATH.exists() or REPORT_PATH.exists():
        raise FileExistsError(
            "Audit outputs already exist. Refusing to overwrite."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("FIRESYNC | MARCH ARCHIVE ATTRIBUTE COMPARISON")
    print("=" * 72)

    archive = prepare(
        pd.read_parquet(ARCHIVE_PATH), "Archive"
    )

    march = prepare(
        pd.read_parquet(MARCH_PATH), "March"
    )

    archive = archive.loc[
        (archive["acq_datetime_utc"] >= "2025-03-01T00:00:00Z")
        & (archive["acq_datetime_utc"] < "2025-04-01T00:00:00Z")
    ].copy()

    march_products = set(march["source_product"].dropna().unique())

    archive["mapped_march_product"] = (
        archive["source_product"].map(PRODUCT_MAP)
    )

    if archive["mapped_march_product"].isna().any():
        raise ValueError("An archive product has no March product mapping.")

    archive_columns = list(dict.fromkeys(
        KEYS
        + ["observation_id", "source_product", "source_file",
           "source_row", "source_archive", "mapped_march_product"]
        + NUMERIC_FIELDS
        + TEXT_FIELDS
    ))

    march_columns = list(dict.fromkeys(
        KEYS
        + ["observation_id", "source_product", "source_file", "source_row"]
        + NUMERIC_FIELDS
        + TEXT_FIELDS
    ))

    merged = archive[archive_columns].merge(
        march[march_columns],
        on=KEYS,
        how="outer",
        suffixes=("_archive", "_march"),
        indicator=True,
        validate="one_to_one",
    )

    if not (merged["_merge"] == "both").all():
        raise ValueError(
            "The key sets differ. Recheck the overlap audit before "
            "comparing attributes."
        )

    if len(merged) != len(march) or len(merged) != len(archive):
        raise ValueError("Unexpected row count after one-to-one merge.")

    comparisons = {}

    for field in NUMERIC_FIELDS:
        comparisons[field] = compare_numeric(
            merged[f"{field}_archive"],
            merged[f"{field}_march"],
        )

    for field in TEXT_FIELDS:
        comparisons[field] = compare_text(
            merged[f"{field}_archive"],
            merged[f"{field}_march"],
        )

    comparisons["mapped_source_product"] = (
        merged["mapped_march_product"]
        .astype("string")
        .eq(merged["source_product_march"].astype("string"))
        .to_numpy()
    )

    detail = pd.DataFrame({
        "archive_observation_id": merged["observation_id_archive"],
        "march_observation_id": merged["observation_id_march"],
        "archive_source_file": merged["source_file_archive"],
        "march_source_file": merged["source_file_march"],
        "archive_source_product": merged["source_product_archive"],
        "march_source_product": merged["source_product_march"],
        "satellite_platform": merged["satellite_platform"],
        "acq_datetime_utc": merged["acq_datetime_utc"],
        "latitude_key": merged["latitude_key"],
        "longitude_key": merged["longitude_key"],
    })

    field_summary = {}

    for field, equal in comparisons.items():
        detail[f"{field}_equal"] = equal

        field_summary[field] = {
            "equal_records": int(equal.sum()),
            "different_records": int((~equal).sum()),
        }

    comparison_columns = list(comparisons.keys())

    detail["all_compared_fields_equal"] = detail[
        [f"{field}_equal" for field in comparison_columns]
    ].all(axis=1)

    detail.to_csv(DETAIL_PATH, index=False)

    report = {
        "project": "FIRESYNC",
        "audit": "March archive and area-API attribute comparison",
        "matched_rows": int(len(merged)),
        "coordinate_precision": 6,
        "numeric_tolerance": {
            "absolute_tolerance": ATOL,
            "relative_tolerance": 0,
        },
        "all_compared_fields_equal_records": int(
            detail["all_compared_fields_equal"].sum()
        ),
        "records_with_at_least_one_difference": int(
            (~detail["all_compared_fields_equal"]).sum()
        ),
        "field_comparison": field_summary,
        "product_mapping": PRODUCT_MAP,
        "march_products_found": sorted(march_products),
        "interpretation": (
            "Comparison is between records sharing the same platform, "
            "UTC timestamp, and coordinates rounded to six decimals. "
            "Numeric fields use the stated absolute tolerance. "
            "Equality does not establish identical processing lineage "
            "or prove a confirmed fire event."
        ),
        "outputs": {
            "detail_csv": str(DETAIL_PATH.relative_to(ROOT)),
            "report_json": str(REPORT_PATH.relative_to(ROOT)),
        },
    }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nATTRIBUTE COMPARISON")
    print("-" * 72)

    for field, result in field_summary.items():
        print(
            f"{field:35} "
            f"equal={result['equal_records']:3} "
            f"different={result['different_records']:3}"
        )

    print("-" * 72)
    print(
        "All fields equal:",
        report["all_compared_fields_equal_records"],
        "/",
        len(merged),
    )
    print(
        "Records with differences:",
        report["records_with_at_least_one_difference"],
    )

    print(f"\nDetail CSV: {DETAIL_PATH}")
    print(f"Report JSON: {REPORT_PATH}")


if __name__ == "__main__":
    main()
