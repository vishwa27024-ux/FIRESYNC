
from pathlib import Path
from io import BytesIO
import hashlib
import json
import zipfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "data" / "archive_downloads" / "nasa_firms" / "incoming"
OUTPUT_DIR = ROOT / "data" / "interim"

OUTPUT_PARQUET = OUTPUT_DIR / "firms_archive_raw_2021_2025.parquet"
OUTPUT_REPORT = OUTPUT_DIR / "firms_archive_import_report.json"

EXPECTED_ARCHIVES = {
    "DL_FIRE_M-C61_839652.zip": {
        "request_id": "839652",
        "source_product": "MODIS_C61_ARCHIVE",
        "sensor": "MODIS",
        "satellite": "AQUA_TERRA",
        "instrument": "MODIS",
    },
    "DL_FIRE_J1V-C2_839653.zip": {
        "request_id": "839653",
        "source_product": "VIIRS_NOAA20_C2_ARCHIVE",
        "sensor": "VIIRS",
        "satellite": "N20",
        "instrument": "VIIRS",
    },
    "DL_FIRE_SV-C2_839654.zip": {
        "request_id": "839654",
        "source_product": "VIIRS_SNPP_C2_ARCHIVE",
        "sensor": "VIIRS",
        "satellite": "SNPP",
        "instrument": "SNPP",
    },
    "DL_FIRE_J2V-C2_839655.zip": {
        "request_id": "839655",
        "source_product": "VIIRS_NOAA21_C2_ARCHIVE",
        "sensor": "VIIRS",
        "satellite": "N21",
        "instrument": "VIIRS",
    },
}

REQUIRED_COLUMNS = {
    "latitude",
    "longitude",
    "brightness",
    "scan",
    "track",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "version",
    "bright_t31",
    "frp",
    "daynight",
    "type",
}

WEST, SOUTH, EAST, NORTH = 76.15, 11.55, 76.91, 11.96


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def validate_archive(path, metadata):
    if not path.exists():
        raise FileNotFoundError(f"Archive missing: {path}")

    archive_hash = sha256_file(path)

    with zipfile.ZipFile(path, "r") as archive:
        bad_member = archive.testzip()

        if bad_member is not None:
            raise ValueError(
                f"ZIP integrity failure in {path.name}: {bad_member}"
            )

        csv_members = [
            name for name in archive.namelist()
            if name.lower().endswith(".csv")
            and not name.startswith("__MACOSX/")
        ]

        if len(csv_members) != 1:
            raise ValueError(
                f"{path.name}: expected exactly one CSV, "
                f"found {len(csv_members)}"
            )

        member = csv_members[0]

        with archive.open(member) as csv_file:
            frame = pd.read_csv(csv_file)

# Preserve native confidence values without mixing numeric and text types.
    frame["confidence"] = frame["confidence"].astype("string")

    missing = REQUIRED_COLUMNS - set(frame.columns)

    if missing:
        raise ValueError(
            f"{path.name}: missing columns: {sorted(missing)}"
        )

    # Preserve original CSV fields and add explicit provenance.
    frame["source_product"] = metadata["source_product"]
    frame["source_file"] = f"{path.name}::{member}"
    frame["source_row"] = range(len(frame))
    frame["archive_request_id"] = metadata["request_id"]
    frame["source_archive"] = path.name
    frame["source_archive_sha256"] = archive_hash

    # Basic coordinate validation.
    frame["latitude"] = pd.to_numeric(
        frame["latitude"], errors="coerce"
    )
    frame["longitude"] = pd.to_numeric(
        frame["longitude"], errors="coerce"
    )

    invalid_coordinates = (
        frame["latitude"].isna()
        | frame["longitude"].isna()
        | ~frame["latitude"].between(-90, 90)
        | ~frame["longitude"].between(-180, 180)
    )

    if invalid_coordinates.any():
        raise ValueError(
            f"{path.name}: {int(invalid_coordinates.sum())} "
            "invalid coordinate rows"
        )

    # Validate product identity.
    satellites = set(
        frame["satellite"].astype(str).str.upper().str.strip()
    )

    instruments = set(
        frame["instrument"].astype(str).str.upper().str.strip()
    )

    expected_satellite = metadata["satellite"]
    expected_instrument = metadata["instrument"]

    if expected_satellite == "AQUA_TERRA":
        if not satellites.issubset({"A", "T", "AQUA", "TERRA"}):
            raise ValueError(
                f"{path.name}: unexpected MODIS satellite values "
                f"{sorted(satellites)}"
            )
    elif satellites != {expected_satellite}:
        raise ValueError(
            f"{path.name}: expected satellite {expected_satellite}, "
            f"found {sorted(satellites)}"
        )

    if instruments != {expected_instrument}:
        raise ValueError(
            f"{path.name}: unexpected instruments {sorted(instruments)}"
        )

    # Validate date and UTC HHMM fields.
    dates = pd.to_datetime(
        frame["acq_date"], errors="coerce"
    )

    times = pd.to_numeric(
        frame["acq_time"], errors="coerce"
    )

    invalid_time = (
        times.isna()
        | (times < 0)
        | (times > 2359)
        | (times % 100 >= 60)
    )

    invalid_date = dates.isna()

    if invalid_date.any() or invalid_time.any():
        raise ValueError(
            f"{path.name}: invalid dates={int(invalid_date.sum())}, "
            f"invalid UTC times={int(invalid_time.sum())}"
        )

    # Check study acquisition rectangle separately from global coordinates.
    inside_study_rectangle = (
        frame["longitude"].between(WEST, EAST)
        & frame["latitude"].between(SOUTH, NORTH)
    )

    summary = {
        "archive": path.name,
        "request_id": metadata["request_id"],
        "source_product": metadata["source_product"],
        "csv_member": member,
        "sha256": archive_hash,
        "records": int(len(frame)),
        "unique_observation_dates": int(dates.dt.date.nunique()),
        "date_min": str(dates.min().date()),
        "date_max": str(dates.max().date()),
        "invalid_coordinates": int(invalid_coordinates.sum()),
        "outside_study_rectangle": int((~inside_study_rectangle).sum()),
        "satellite_values": sorted(satellites),
        "instrument_values": sorted(instruments),
        "status": "validated",
    }

    return frame, summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PARQUET.exists() or OUTPUT_REPORT.exists():
        raise FileExistsError(
            "Output already exists. Refusing to overwrite:\n"
            f"{OUTPUT_PARQUET}\n{OUTPUT_REPORT}"
        )

    frames = []
    summaries = []

    print("=" * 72)
    print("FIRESYNC | NASA FIRMS ARCHIVE IMPORT")
    print("=" * 72)

    for archive_name, metadata in EXPECTED_ARCHIVES.items():
        archive_path = INPUT_DIR / archive_name

        print(f"\nValidating: {archive_name}")

        frame, summary = validate_archive(
            archive_path,
            metadata,
        )

        frames.append(frame)
        summaries.append(summary)

        print(f"  Records: {summary['records']}")
        print(f"  Date range: {summary['date_min']} to {summary['date_max']}")
        print(f"  SHA-256: {summary['sha256']}")
        print("  Status: PASS")

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    combined["source_row"] = combined["source_row"].astype("int64")

    expected_ids = combined[
        ["source_file", "source_row"]
    ].drop_duplicates()

    if len(expected_ids) != len(combined):
        raise ValueError(
            "Source provenance collision detected. "
            "The combination of source_file and source_row is not unique."
        )

    report = {
        "project": "FIRESYNC",
        "dataset": "NASA FIRMS archive raw observations",
        "period_requested": "2021-01-01 to 2025-12-31",
        "quality_label": (
            "Archive download; processing-quality classification "
            "not independently established by this importer."
        ),
        "study_rectangle": {
            "west": WEST,
            "south": SOUTH,
            "east": EAST,
            "north": NORTH,
            "description": (
                "Bandipur landscape acquisition rectangle; "
                "not an official protected-area boundary."
            ),
        },
        "archive_count": len(summaries),
        "total_records": int(len(combined)),
        "archives": summaries,
        "outputs": {
            "raw_parquet": str(OUTPUT_PARQUET.relative_to(ROOT)),
            "report": str(OUTPUT_REPORT.relative_to(ROOT)),
        },
    }

    # Write to temporary paths first to avoid leaving partial outputs.
    temporary_parquet = OUTPUT_PARQUET.with_suffix(".tmp.parquet")
    temporary_report = OUTPUT_REPORT.with_suffix(".tmp.json")

    try:
        combined.to_parquet(
            temporary_parquet,
            index=False,
            engine="pyarrow",
        )

        with temporary_report.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

        temporary_parquet.replace(OUTPUT_PARQUET)
        temporary_report.replace(OUTPUT_REPORT)

    except Exception:
        temporary_parquet.unlink(missing_ok=True)
        temporary_report.unlink(missing_ok=True)
        raise

    print("\n" + "=" * 72)
    print("IMPORT COMPLETE")
    print("=" * 72)
    print(f"Total records: {len(combined):,}")
    print(f"Raw Parquet: {OUTPUT_PARQUET}")
    print(f"Report: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
