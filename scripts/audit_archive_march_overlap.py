
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

ARCHIVE_PATH = (
    ROOT / "data" / "processed"
    / "firms_archive_normalized_2021_2025.parquet"
)

MARCH_PATH = (
    ROOT / "data" / "processed"
    / "march_2025_normalized_observations.parquet"
)

OUTPUT_DIR = ROOT / "data" / "interim"

REPORT_PATH = OUTPUT_DIR / "archive_march_overlap_report.json"
DETAIL_PATH = OUTPUT_DIR / "archive_march_overlap_keys.csv"

KEY_COLUMNS = [
    "sensor_family",
    "satellite_platform",
    "acq_datetime_utc",
    "latitude_key",
    "longitude_key",
]


def prepare_keys(frame, dataset_name):
    required = [
        "sensor_family",
        "satellite_platform",
        "acq_datetime_utc",
        "latitude",
        "longitude",
        "observation_id",
    ]

    missing = set(required) - set(frame.columns)

    if missing:
        raise ValueError(
            f"{dataset_name} is missing columns: {sorted(missing)}"
        )

    result = frame.copy()

    result["acq_datetime_utc"] = pd.to_datetime(
        result["acq_datetime_utc"],
        utc=True,
        errors="coerce",
    )

    result["latitude"] = pd.to_numeric(
        result["latitude"], errors="coerce"
    )

    result["longitude"] = pd.to_numeric(
        result["longitude"], errors="coerce"
    )

    invalid = (
        result["acq_datetime_utc"].isna()
        | result["latitude"].isna()
        | result["longitude"].isna()
    )

    if invalid.any():
        raise ValueError(
            f"{dataset_name}: {int(invalid.sum())} rows have "
            "invalid timestamp or coordinates"
        )

    result["latitude_key"] = result["latitude"].round(6)
    result["longitude_key"] = result["longitude"].round(6)

    result["sensor_family"] = (
        result["sensor_family"].astype("string")
    )

    result["satellite_platform"] = (
        result["satellite_platform"].astype("string")
    )

    return result


def grouped_key_counts(frame, count_name):
    return (
        frame.groupby(KEY_COLUMNS, dropna=False)
        .size()
        .rename(count_name)
        .reset_index()
    )


def main():
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(ARCHIVE_PATH)

    if not MARCH_PATH.exists():
        raise FileNotFoundError(MARCH_PATH)

    if REPORT_PATH.exists() or DETAIL_PATH.exists():
        raise FileExistsError(
            "Audit outputs already exist. Refusing to overwrite:\n"
            f"{REPORT_PATH}\n{DETAIL_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("FIRESYNC | ARCHIVE VS MARCH 2025 OVERLAP AUDIT")
    print("=" * 72)

    archive_all = pd.read_parquet(ARCHIVE_PATH)
    march = pd.read_parquet(MARCH_PATH)

    archive_all = prepare_keys(archive_all, "Archive")
    march = prepare_keys(march, "March area-API")

    # Restrict the archive dataset to March 2025.
    archive = archive_all.loc[
        (archive_all["acq_datetime_utc"] >= "2025-03-01T00:00:00Z")
        & (archive_all["acq_datetime_utc"] < "2025-04-01T00:00:00Z")
    ].copy()

    # Confirm the comparison dataset contains March 2025 only.
    march_months = march["acq_datetime_utc"].dt.strftime("%Y-%m").unique()

    if set(march_months) != {"2025-03"}:
        raise ValueError(
            "Expected the March dataset to contain March 2025 only. "
            f"Found: {sorted(march_months)}"
        )

    print(f"\nArchive total observations: {len(archive_all):,}")
    print(f"Archive March 2025 observations: {len(archive):,}")
    print(f"March area-API observations: {len(march):,}")

    archive_counts = grouped_key_counts(
        archive, "archive_count"
    )

    march_counts = grouped_key_counts(
        march, "march_count"
    )

    comparison = archive_counts.merge(
        march_counts,
        on=KEY_COLUMNS,
        how="outer",
    )

    comparison["archive_count"] = (
        comparison["archive_count"].fillna(0).astype(int)
    )

    comparison["march_count"] = (
        comparison["march_count"].fillna(0).astype(int)
    )

    comparison["matched_record_count"] = comparison[
        ["archive_count", "march_count"]
    ].min(axis=1)

    comparison["archive_unmatched_count"] = (
        comparison["archive_count"]
        - comparison["matched_record_count"]
    )

    comparison["march_unmatched_count"] = (
        comparison["march_count"]
        - comparison["matched_record_count"]
    )

    comparison["match_status"] = "no_match"

    comparison.loc[
        (comparison["archive_count"] > 0)
        & (comparison["march_count"] > 0),
        "match_status",
    ] = "shared_spatiotemporal_key"

    comparison.loc[
        (comparison["archive_count"] > 1)
        | (comparison["march_count"] > 1),
        "match_status",
    ] = "duplicate_key_requires_review"

    matched_keys = comparison.loc[
        (comparison["archive_count"] > 0)
        & (comparison["march_count"] > 0)
    ]

    matched_records = int(
        comparison["matched_record_count"].sum()
    )

    archive_unmatched = int(
        comparison["archive_unmatched_count"].sum()
    )

    march_unmatched = int(
        comparison["march_unmatched_count"].sum()
    )

    duplicate_archive_keys = int(
        (archive_counts["archive_count"] > 1).sum()
    )

    duplicate_march_keys = int(
        (march_counts["march_count"] > 1).sum()
    )

    report = {
        "project": "FIRESYNC",
        "audit": "Archive versus March 2025 area-API overlap",
        "comparison_method": {
            "fields": KEY_COLUMNS,
            "coordinate_precision": 6,
            "timestamp": "UTC",
            "matching_rule": (
                "Same sensor family, satellite platform, UTC acquisition "
                "timestamp, and latitude/longitude rounded to six decimals."
            ),
        },
        "counts": {
            "archive_all_periods": int(len(archive_all)),
            "archive_march_2025": int(len(archive)),
            "march_area_api": int(len(march)),
            "shared_spatiotemporal_keys": int(len(matched_keys)),
            "matched_record_count": matched_records,
            "archive_unmatched_records": archive_unmatched,
            "march_unmatched_records": march_unmatched,
            "archive_duplicate_keys": duplicate_archive_keys,
            "march_duplicate_keys": duplicate_march_keys,
            "comparison_key_groups": int(len(comparison)),
        },
        "interpretation": (
            "A shared key is an exact spatiotemporal key match at the "
            "specified coordinate precision. It is not proof of a "
            "confirmed fire event. No observations were deleted."
        ),
        "outputs": {
            "detail_csv": str(DETAIL_PATH.relative_to(ROOT)),
            "report_json": str(REPORT_PATH.relative_to(ROOT)),
        },
    }

    comparison.to_csv(DETAIL_PATH, index=False)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n" + "-" * 72)
    print("OVERLAP AUDIT RESULTS")
    print("-" * 72)
    print(f"Shared keys: {len(matched_keys):,}")
    print(f"Matched record count: {matched_records:,}")
    print(f"Archive unmatched: {archive_unmatched:,}")
    print(f"March unmatched: {march_unmatched:,}")
    print(f"Archive duplicate keys: {duplicate_archive_keys:,}")
    print(f"March duplicate keys: {duplicate_march_keys:,}")

    print("\nNo records have been deleted or merged.")
    print(f"\nDetail CSV: {DETAIL_PATH}")
    print(f"Report JSON: {REPORT_PATH}")


if __name__ == "__main__":
    main()
