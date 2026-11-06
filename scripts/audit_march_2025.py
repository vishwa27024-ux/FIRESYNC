
from pathlib import Path
import hashlib
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MANIFEST = (
    ROOT / "data" / "interim"
    / "march_2025_full_month_manifest.json"
)

OUTPUT = ROOT / "data" / "interim"

PROCESSED = ROOT / "data" / "processed"

PRODUCTS = [
    "MODIS_SP",
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_SP",
    "VIIRS_NOAA21_SP",
]

START = pd.Timestamp("2025-03-01")
END = pd.Timestamp("2025-03-31")

BBOX = {
    "west": 76.15,
    "south": 11.55,
    "east": 76.91,
    "north": 11.96,
}


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    print("=" * 65)
    print("FIRESYNC | MARCH 2025 DATA QUALITY AUDIT")
    print("=" * 65)

    with MANIFEST.open(encoding="utf-8") as file:
        manifest = json.load(file)

    if manifest.get("status") != "complete":
        raise RuntimeError("Acquisition manifest is not complete.")

    entries = manifest["entries"]

    expected = {
        (product, start)
        for product in PRODUCTS
        for start in [
            "2025-03-01",
            "2025-03-06",
            "2025-03-11",
            "2025-03-16",
            "2025-03-21",
            "2025-03-26",
            "2025-03-31",
        ]
    }

    actual = {
        (entry["source"], entry["start_date"])
        for entry in entries
    }

    if actual != expected or len(entries) != 28:
        raise RuntimeError(
            "Manifest does not contain exactly 28 unique expected chunks."
        )

    frames = []
    product_results = []
    failures = []

    for product in PRODUCTS:
        print(f"\nInspecting {product}")

        product_frames = []

        product_entries = [
            item for item in entries
            if item["source"] == product
        ]

        for entry in product_entries:
            csv_path = ROOT / entry["file"]

            if not csv_path.is_file():
                raise FileNotFoundError(csv_path)

            actual_hash = sha256_file(csv_path)

            if actual_hash != entry["sha256"]:
                raise RuntimeError(
                    f"SHA-256 mismatch: {csv_path.name}"
                )

            data = pd.read_csv(csv_path)

            if len(data) != entry["record_count"]:
                raise RuntimeError(
                    f"Record count mismatch: {csv_path.name}"
                )

            required = {
                "latitude",
                "longitude",
                "acq_date",
                "acq_time",
                "satellite",
                "instrument",
                "confidence",
                "version",
                "scan",
                "track",
            }

            missing = required - set(data.columns)

            if missing:
                raise RuntimeError(
                    f"{product} missing columns: {sorted(missing)}"
                )

            data["source_product"] = product
            data["source_file"] = csv_path.name

            product_frames.append(data)

        if not product_frames:
            raise RuntimeError(f"No chunks for {product}")

        product_data = pd.concat(
            product_frames,
            ignore_index=True,
            sort=False,
        )

        latitude = pd.to_numeric(
            product_data["latitude"], errors="coerce"
        )

        longitude = pd.to_numeric(
            product_data["longitude"], errors="coerce"
        )

        dates = pd.to_datetime(
            product_data["acq_date"], errors="coerce"
        )

        valid_coordinates = (
            latitude.between(-90, 90)
            & longitude.between(-180, 180)
        )

        inside_rectangle = (
            latitude.between(BBOX["south"], BBOX["north"])
            & longitude.between(BBOX["west"], BBOX["east"])
        )

        valid_dates = dates.between(START, END)

        exact_duplicates = int(
            product_data.duplicated(
                subset=[
                    col for col in product_data.columns
                    if col not in ["source_file"]
                ]
            ).sum()
        )

        invalid_coordinates = int(
            (~valid_coordinates).sum()
        )

        outside_rectangle = int(
            (~inside_rectangle).sum()
        )

        invalid_dates = int(dates.isna().sum())

        outside_period = int((~valid_dates).sum())

        result = {
            "product": product,
            "records": len(product_data),
            "invalid_coordinates": invalid_coordinates,
            "outside_rectangle": outside_rectangle,
            "invalid_dates": invalid_dates,
            "outside_period": outside_period,
            "exact_duplicate_rows": exact_duplicates,
            "first_date": (
                dates.min().date().isoformat()
                if dates.notna().any() else None
            ),
            "last_date": (
                dates.max().date().isoformat()
                if dates.notna().any() else None
            ),
            "unique_observation_dates": int(
                dates.nunique()
            ),
        }

        product_results.append(result)
        frames.append(product_data)

        print(f"Records: {result['records']}")
        print(f"Invalid coordinates: {invalid_coordinates}")
        print(f"Outside acquisition rectangle: {outside_rectangle}")
        print(f"Invalid dates: {invalid_dates}")
        print(f"Outside March: {outside_period}")
        print(f"Exact duplicate rows: {exact_duplicates}")

        if (
            invalid_coordinates
            or outside_rectangle
            or invalid_dates
            or outside_period
        ):
            failures.append(product)

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


    # Preserve native confidence values without mixing types.
    # MODIS: numerical confidence; VIIRS: categorical confidence.
    combined["confidence"] = combined["confidence"].astype("string")


    combined["acq_date"] = pd.to_datetime(
        combined["acq_date"],
        errors="coerce",
    )

    combined["sensor_family"] = combined[
        "source_product"
    ].map(
        lambda product: (
            "MODIS" if product == "MODIS_SP" else "VIIRS"
        )
    )

    daily = (
        combined.groupby(
            ["acq_date", "source_product"]
        )
        .size()
        .unstack(fill_value=0)
    )

    calendar = pd.date_range(START, END, freq="D")

    daily = daily.reindex(
        index=calendar,
        columns=PRODUCTS,
        fill_value=0,
    )

    daily.index.name = "acq_date"

    summary = pd.DataFrame(product_results)

    duplicate_product_dates = combined.duplicated(
        subset=[
            "source_product",
            "latitude",
            "longitude",
            "acq_date",
            "acq_time",
            "satellite",
        ]
    ).sum()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    daily.to_csv(
        OUTPUT / "march_2025_daily_counts.csv"
    )

    summary.to_csv(
        OUTPUT / "march_2025_product_quality.csv",
        index=False,
    )

    # Common superset schema: native sensor fields are retained.
    parquet_path = (
        PROCESSED / "march_2025_landscape_observations.parquet"
    )

    if parquet_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite {parquet_path}"
        )

    combined.to_parquet(
        parquet_path,
        index=False,
    )

    report = {
        "project": "FIRESYNC",
        "period": "2025-03-01 to 2025-03-31",
        "region": (
            "Bandipur landscape rectangle; "
            "not an official reserve boundary."
        ),
        "manifest_status": manifest["status"],
        "product_window_combinations": len(entries),
        "total_observations": len(combined),
        "invalid_products": failures,
        "potential_duplicate_product_date_coordinate_rows": int(
            duplicate_product_dates
        ),
        "per_product": product_results,
        "daily_count_file": "march_2025_daily_counts.csv",
        "quality_file": "march_2025_product_quality.csv",
        "parquet_file": str(parquet_path.relative_to(ROOT)),
        "raw_csv_files_modified": False,
    }

    report_path = OUTPUT / "march_2025_quality_report.json"

    if report_path.exists():
        raise FileExistsError(report_path)

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("FINAL MONTHLY AUDIT")
    print("=" * 65)

    print("Product-window combinations:", len(entries))
    print("Total observations:", len(combined))
    print("Potential duplicate product/date/location rows:",
          duplicate_product_dates)

    print("\nPRODUCT SUMMARY")
    print(summary.to_string(index=False))

    print("\nOUTPUT FILES")
    print(daily)
    print("Parquet:", parquet_path)
    print("Report:", report_path)

    if failures:
        print("\nQUALITY ISSUES FOUND:", failures)
        raise SystemExit(1)

    print("\nINITIAL QUALITY CHECKS PASSED.")
    print("Raw NASA CSV files remain unchanged.")
    print("=" * 65)


if __name__ == "__main__":
    main()
