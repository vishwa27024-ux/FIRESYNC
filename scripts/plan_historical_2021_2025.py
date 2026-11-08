
import hashlib
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

END_DATE = date(2025, 12, 31)

AREA = {
    "west": 76.15,
    "south": 11.55,
    "east": 76.91,
    "north": 11.96,
}

BBOX = [
    AREA["west"],
    AREA["south"],
    AREA["east"],
    AREA["north"],
]

# Historical start dates selected for this project.
PRODUCT_STARTS = {
    "MODIS_SP": date(2021, 1, 1),
    "VIIRS_SNPP_SP": date(2021, 1, 1),
    "VIIRS_NOAA20_SP": date(2021, 1, 1),
    "VIIRS_NOAA21_SP": date(2023, 3, 1),
}

# Availability snapshot verified using NASA FIRMS on 2026-11-06.
AVAILABILITY = {
    "MODIS_SP": ("2000-11-01", "2026-08-31"),
    "VIIRS_SNPP_SP": ("2012-01-20", "2026-07-31"),
    "VIIRS_NOAA20_SP": ("2018-04-01", "2026-07-31"),
    "VIIRS_NOAA21_SP": ("2023-03-01", "2026-07-31"),
}

OUTPUT_CSV = (
    ROOT / "data" / "interim"
    / "historical_2021_2025_plan.csv"
)

OUTPUT_JSON = (
    ROOT / "data" / "interim"
    / "historical_2021_2025_plan.json"
)


def build_monthly_windows(start_date, end_date):
    """Create sequential windows of at most five days per month."""

    windows = []

    current_month = date(
        start_date.year,
        start_date.month,
        1,
    )

    while current_month <= end_date:
        if current_month.month == 12:
            next_month = date(current_month.year + 1, 1, 1)
        else:
            next_month = date(
                current_month.year,
                current_month.month + 1,
                1,
            )

        month_end = min(
            next_month - timedelta(days=1),
            end_date,
        )

        current = max(current_month, start_date)

        while current <= month_end:
            day_range = min(
                5,
                (month_end - current).days + 1,
            )

            window_end = current + timedelta(days=day_range - 1)

            windows.append({
                "start": current.isoformat(),
                "end": window_end.isoformat(),
                "day_range": day_range,
            })

            current = window_end + timedelta(days=1)

        current_month = next_month

    return windows


def get_paths(source, window):
    folder = (
        "modis"
        if source.startswith("MODIS")
        else "viirs"
    )

    filename = (
        f"{source}_{window['start']}_to_"
        f"{window['end']}.csv"
    )

    csv_path = (
        ROOT / "data" / "raw"
        / folder / source / filename
    )

    return csv_path, csv_path.with_suffix(".json")


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def validate_existing_chunk(source, window):
    csv_path, metadata_path = get_paths(source, window)

    if not csv_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "CSV and metadata must both exist."
        )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    expected = {
        "source": source,
        "start_date": window["start"],
        "end_date": window["end"],
        "day_range": window["day_range"],
    }

    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Metadata mismatch: {key}"
            )

    actual_bbox = metadata.get(
        "bounding_box_west_south_east_north"
    )

    if (
        not isinstance(actual_bbox, list)
        or len(actual_bbox) != 4
        or not all(
            math.isclose(
                float(a),
                float(b),
                abs_tol=1e-8,
            )
            for a, b in zip(actual_bbox, BBOX)
        )
    ):
        raise ValueError("Bounding box mismatch.")

    data = pd.read_csv(csv_path)

    required = {
        "latitude",
        "longitude",
        "acq_date",
    }

    if not required.issubset(data.columns):
        raise ValueError("Required CSV columns missing.")

    if len(data) != int(metadata.get("record_count", -1)):
        raise ValueError("CSV record count mismatch.")

    if not data.empty:
        lat = pd.to_numeric(
            data["latitude"], errors="coerce"
        )
        lon = pd.to_numeric(
            data["longitude"], errors="coerce"
        )
        dates = pd.to_datetime(
            data["acq_date"], errors="coerce"
        )

        if lat.isna().any() or lon.isna().any():
            raise ValueError("Invalid coordinates.")

        if (
            not lat.between(-90, 90).all()
            or not lon.between(-180, 180).all()
        ):
            raise ValueError("Coordinates out of range.")

        if dates.isna().any():
            raise ValueError("Invalid acquisition dates.")

        if not dates.between(
            pd.Timestamp(window["start"]),
            pd.Timestamp(window["end"]),
        ).all():
            raise ValueError("Date outside requested window.")

    return {
        "record_count": len(data),
        "file_bytes": csv_path.stat().st_size,
        "sha256": sha256_file(csv_path),
    }


def main():
    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    expected_total = 0

    print("=" * 72)
    print("FIRESYNC | HISTORICAL ACQUISITION PLANNER")
    print("=" * 72)

    for source, start_date in PRODUCT_STARTS.items():
        available_start, available_end = AVAILABILITY[source]

        if (
            date.fromisoformat(available_start) > start_date
            or date.fromisoformat(available_end) < END_DATE
        ):
            raise RuntimeError(
                f"{source} does not cover its planned period."
            )

        windows = build_monthly_windows(
            start_date,
            END_DATE,
        )

        expected_total += len(windows)

        print(
            f"\n{source}: {start_date} to {END_DATE}"
        )
        print(f"Expected windows: {len(windows)}")

        for window in windows:
            csv_path, metadata_path = get_paths(
                source,
                window,
            )

            csv_exists = csv_path.is_file()
            metadata_exists = metadata_path.is_file()

            result = {
                "source": source,
                **window,
                "csv_file": csv_path.relative_to(ROOT).as_posix(),
                "metadata_file": metadata_path.relative_to(ROOT).as_posix(),
                "record_count": None,
                "file_bytes": None,
                "sha256": None,
                "status": None,
                "error": None,
            }

            if not csv_exists and not metadata_exists:
                result["status"] = "missing_download"

            elif csv_exists != metadata_exists:
                result["status"] = "incomplete_pair"
                result["error"] = (
                    "Only CSV or metadata exists. "
                    "Manual inspection required."
                )

            else:
                try:
                    validation = validate_existing_chunk(
                        source,
                        window,
                    )

                    result.update(validation)

                    result["status"] = (
                        "existing_valid_empty"
                        if validation["record_count"] == 0
                        else "existing_valid"
                    )

                except Exception as error:
                    result["status"] = "invalid_existing"
                    result["error"] = str(error)

            rows.append(result)

    plan = pd.DataFrame(rows)
    plan.to_csv(OUTPUT_CSV, index=False)

    counts = (
        plan["status"]
        .value_counts()
        .to_dict()
    )

    summary = {
        "project": "FIRESYNC",
        "period_end": END_DATE.isoformat(),
        "availability_snapshot_date": "2026-11-06",
        "bounding_box": AREA,
        "maximum_days_per_request": 5,
        "expected_combinations": expected_total,
        "status_counts": counts,
        "valid_existing_count": int(
            counts.get("existing_valid", 0)
        ),
        "valid_empty_count": int(
            counts.get("existing_valid_empty", 0)
        ),
        "missing_download_count": int(
            counts.get("missing_download", 0)
        ),
        "invalid_existing_count": int(
            counts.get("invalid_existing", 0)
        ),
        "incomplete_pair_count": int(
            counts.get("incomplete_pair", 0)
        ),
        "new_download_candidates": int(
            counts.get("missing_download", 0)
        ),
        "planning_only": True,
        "no_nasa_data_requests_made": True,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    OUTPUT_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("PLANNING SUMMARY")
    print("=" * 72)

    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")

    print(f"\nExpected combinations: {expected_total}")
    print(f"CSV plan: {OUTPUT_CSV}")
    print(f"JSON summary: {OUTPUT_JSON}")
    print("\nNo NASA download requests were made.")


if __name__ == "__main__":
    main()
