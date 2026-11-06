
from pathlib import Path
from datetime import date, timedelta, datetime, timezone
import hashlib
import json
import math
import time

import pandas as pd

from src.data.firms_client import (
    SUPPORTED_PRODUCTS,
    fetch_data_availability,
    download_area_csv,
)


ROOT = Path(__file__).resolve().parents[1]

START_DATE = date(2025, 3, 1)
END_DATE = date(2025, 3, 31)

# Bandipur landscape acquisition rectangle.
# This is NOT an official protected-area boundary.
AREA = {
    "west": 76.15,
    "south": 11.55,
    "east": 76.91,
    "north": 11.96,
}

PRODUCTS = sorted(SUPPORTED_PRODUCTS)

MANIFEST = (
    ROOT / "data" / "interim"
    / "march_2025_full_month_manifest.json"
)

MAX_RETRIES = 3
REQUEST_DELAY = 1.0


def build_windows():
    """Create inclusive date windows of at most five days."""
    windows = []
    current = START_DATE

    while current <= END_DATE:
        days = min(5, (END_DATE - current).days + 1)
        end = current + timedelta(days=days - 1)

        windows.append({
            "start": current.isoformat(),
            "end": end.isoformat(),
            "day_range": days,
        })

        current = end + timedelta(days=1)

    return windows


def get_paths(source, window):
    folder = "modis" if source.startswith("MODIS") else "viirs"

    filename = (
        f"{source}_{window['start']}_to_{window['end']}.csv"
    )

    csv_path = (
        ROOT / "data" / "raw" / folder / source / filename
    )

    return csv_path, csv_path.with_suffix(".json")


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def validate_chunk(source, window):
    """Validate an existing original CSV and its metadata."""
    csv_path, metadata_path = get_paths(source, window)

    if not csv_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"CSV or metadata missing: {csv_path.name}"
        )

    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    expected_bbox = [
        AREA["west"],
        AREA["south"],
        AREA["east"],
        AREA["north"],
    ]

    checks = {
        "source": source,
        "start_date": window["start"],
        "end_date": window["end"],
        "day_range": window["day_range"],
    }

    for key, expected in checks.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Metadata mismatch: {key}")

    actual_bbox = metadata.get(
        "bounding_box_west_south_east_north"
    )

    if (
        not isinstance(actual_bbox, list)
        or len(actual_bbox) != 4
        or not all(
            math.isclose(float(a), float(b), abs_tol=1e-8)
            for a, b in zip(actual_bbox, expected_bbox)
        )
    ):
        raise ValueError("Bounding box metadata mismatch.")

    data = pd.read_csv(csv_path)

    required = {"latitude", "longitude", "acq_date"}

    if not required.issubset(data.columns):
        raise ValueError("Required CSV columns are missing.")

    if len(data) != int(metadata.get("record_count", -1)):
        raise ValueError("CSV record count mismatch.")

    if not data.empty:
        lat = pd.to_numeric(data["latitude"], errors="coerce")
        lon = pd.to_numeric(data["longitude"], errors="coerce")
        dates = pd.to_datetime(data["acq_date"], errors="coerce")

        if lat.isna().any() or lon.isna().any():
            raise ValueError("Invalid coordinates detected.")

        if not lat.between(-90, 90).all():
            raise ValueError("Latitude outside valid range.")

        if not lon.between(-180, 180).all():
            raise ValueError("Longitude outside valid range.")

        if dates.isna().any():
            raise ValueError("Invalid acquisition date detected.")

        if not dates.between(
            pd.Timestamp(window["start"]),
            pd.Timestamp(window["end"]),
        ).all():
            raise ValueError("Observation outside requested dates.")

    return {
        "file": csv_path.relative_to(ROOT).as_posix(),
        "metadata_file": metadata_path.relative_to(ROOT).as_posix(),
        "record_count": len(data),
        "file_bytes": csv_path.stat().st_size,
        "sha256": sha256_file(csv_path),
        "downloaded_at_utc": metadata.get("downloaded_at_utc"),
    }


def check_availability():
    print("\nChecking NASA FIRMS availability...")

    availability = fetch_data_availability("all")

    for source in PRODUCTS:
        rows = availability[
            availability["data_id"].astype(str).str.strip() == source
        ]

        if rows.empty:
            raise RuntimeError(
                f"NASA did not report availability for {source}."
            )

        row = rows.iloc[0]

        minimum = pd.Timestamp(row["min_date"]).date()
        maximum = pd.Timestamp(row["max_date"]).date()

        print(f"{source}: {minimum} to {maximum}")

        if minimum > START_DATE or maximum < END_DATE:
            raise RuntimeError(
                f"{source} does not cover March 2025."
            )

    print("Product date-range coverage verified.")
    print("This does not prove observations exist on every day.")


def retryable(error):
    message = str(error).lower()

    return any(term in message for term in (
        "http 429",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
        "network connection failed",
        "could not connect",
        "timed out",
    ))


def download_with_retry(source, window):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  API attempt {attempt}/{MAX_RETRIES}")

            return download_area_csv(
                source=source,
                start_date=window["start"],
                day_range=window["day_range"],
                **AREA,
            )

        except Exception as error:
            if not retryable(error) or attempt == MAX_RETRIES:
                raise

            delay = 2 ** attempt
            print(f"  Temporary error. Retrying in {delay}s.")
            time.sleep(delay)


def save_manifest(entries, failures, status):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    document = {
        "project": "FIRESYNC",
        "period_start": START_DATE.isoformat(),
        "period_end": END_DATE.isoformat(),
        "status": status,
        "study_region": (
            "Bandipur landscape rectangle; not an official reserve boundary"
        ),
        "crs": "EPSG:4326",
        "bounding_box": AREA,
        "products": PRODUCTS,
        "maximum_days_per_request": 5,
        "expected_combinations": len(PRODUCTS) * len(build_windows()),
        "completed_combinations": len(entries),
        "failed_combinations": len(failures),
        "entries": entries,
        "failures": failures,
        "updated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    temporary = MANIFEST.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(document, indent=2),
        encoding="utf-8",
    )

    temporary.replace(MANIFEST)


def main():
    windows = build_windows()
    expected = len(PRODUCTS) * len(windows)

    print("=" * 65)
    print("FIRESYNC | MARCH 2025 FULL-MONTH ACQUISITION")
    print("=" * 65)

    print(f"Products: {len(PRODUCTS)}")
    print(f"Windows per product: {len(windows)}")
    print(f"Total product-window combinations: {expected}")

    for window in windows:
        print(
            f"{window['start']} to {window['end']} "
            f"({window['day_range']} days)"
        )

    check_availability()

    entries = []
    failures = []

    save_manifest(entries, failures, "in_progress")

    for source in PRODUCTS:
        print("\n" + "-" * 65)
        print(f"PRODUCT: {source}")
        print("-" * 65)

        for window in windows:
            label = f"{window['start']} to {window['end']}"
            csv_path, metadata_path = get_paths(source, window)

            print(f"\nWindow: {label}")

            try:
                csv_exists = csv_path.exists()
                metadata_exists = metadata_path.exists()

                if csv_exists != metadata_exists:
                    raise RuntimeError(
                        "Incomplete existing download. "
                        "Manual inspection required; refusing overwrite."
                    )

                if csv_exists and metadata_exists:
                    result = validate_chunk(source, window)
                    status = "reused_existing"
                    print(
                        f"  Existing file verified: "
                        f"{result['record_count']} records"
                    )

                else:
                    # Never automatically replace our original sample.
                    if window["start"] == "2025-03-01":
                        raise FileNotFoundError(
                            "Original March 1-5 sample missing. "
                            "Refusing automatic replacement."
                        )

                    path, _ = download_with_retry(source, window)

                    result = validate_chunk(source, window)
                    status = "downloaded"

                    print(f"  Saved: {path}")
                    time.sleep(REQUEST_DELAY)

                entries.append({
                    "source": source,
                    "start_date": window["start"],
                    "end_date": window["end"],
                    "day_range": window["day_range"],
                    "status": status,
                    **result,
                })

            except Exception as error:
                print(f"  FAILED: {error}")

                failures.append({
                    "source": source,
                    "start_date": window["start"],
                    "end_date": window["end"],
                    "error": str(error),
                })

            save_manifest(entries, failures, "in_progress")

    complete = len(entries) == expected and not failures
    final_status = "complete" if complete else "partial"

    save_manifest(entries, failures, final_status)

    total_records = sum(
        item["record_count"] for item in entries
    )

    print("\n" + "=" * 65)
    print("MARCH 2025 ACQUISITION SUMMARY")
    print("=" * 65)
    print(f"Expected combinations: {expected}")
    print(f"Completed combinations: {len(entries)}")
    print(f"Failed combinations: {len(failures)}")
    print(f"Total raw observations: {total_records:,}")
    print(f"Manifest status: {final_status}")
    print(f"Manifest: {MANIFEST}")

    if failures:
        for failure in failures:
            print(
                failure["source"],
                failure["start_date"],
                failure["error"],
            )
        raise SystemExit(1)

    print("All date windows downloaded or validated.")
    print("Original five-day sample preserved.")
    print("=" * 65)


if __name__ == "__main__":
    main()
