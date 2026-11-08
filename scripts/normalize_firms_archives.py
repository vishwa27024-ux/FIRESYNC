
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "interim" / "firms_archive_raw_2021_2025.parquet"
OUTPUT = ROOT / "data" / "processed" / "firms_archive_normalized_2021_2025.parquet"
REPORT = ROOT / "data" / "interim" / "firms_archive_normalization_report.json"

PRODUCTS = {
    "MODIS_C61_ARCHIVE": {
        "sensor_family": "MODIS",
        "expected_satellites": {"AQUA", "TERRA"},
    },
    "VIIRS_NOAA20_C2_ARCHIVE": {
        "sensor_family": "VIIRS",
        "expected_satellites": {"N20"},
    },
    "VIIRS_SNPP_C2_ARCHIVE": {
        "sensor_family": "VIIRS",
        "expected_satellites": {"SNPP"},
    },
    "VIIRS_NOAA21_C2_ARCHIVE": {
        "sensor_family": "VIIRS",
        "expected_satellites": {"N21"},
    },
}

PLATFORM_NAMES = {
    "A": "Aqua",
    "AQUA": "Aqua",
    "T": "Terra",
    "TERRA": "Terra",
    "N20": "NOAA-20",
    "N21": "NOAA-21",
    "SNPP": "Suomi-NPP",
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
    "source_product",
    "source_file",
    "source_row",
    "archive_request_id",
    "source_archive",
    "source_archive_sha256",
}


def observation_id(source_product, source_file, source_row):
    raw = f"{source_product}|{source_file}|{source_row}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_utc_datetime(date_values, time_values):
    dates = pd.to_datetime(date_values, errors="coerce").dt.strftime("%Y-%m-%d")

    times = pd.to_numeric(time_values, errors="coerce")

    valid_time = (
        times.notna()
        & times.between(0, 2359)
        & (times % 100 < 60)
    )

    hhmm = times.fillna(-1).astype("int64").astype(str).str.zfill(4)

    valid = dates.notna() & valid_time

    combined = pd.Series(pd.NaT, index=date_values.index, dtype="datetime64[ns, UTC]")

    parsed = pd.to_datetime(
        dates[valid] + " " + hhmm[valid],
        format="%Y-%m-%d %H%M",
        errors="coerce",
        utc=True,
    )

    combined.loc[valid] = parsed

    return combined, hhmm.where(valid, pd.NA), valid


def normalize(raw):
    missing = REQUIRED_COLUMNS - set(raw.columns)

    if missing:
        raise ValueError(f"Raw dataset missing columns: {sorted(missing)}")

    output_frames = []

    for product, metadata in PRODUCTS.items():
        frame = raw.loc[raw["source_product"] == product].copy()

        if frame.empty:
            raise ValueError(f"No observations found for {product}")

        satellite_codes = (
            frame["satellite"].astype("string").str.upper().str.strip()
        )

        if set(satellite_codes.dropna().unique()) != metadata["expected_satellites"]:
            raise ValueError(
                f"Unexpected satellite codes for {product}: "
                f"{sorted(satellite_codes.dropna().unique())}"
            )

        frame["satellite_code"] = satellite_codes

        frame["satellite_platform"] = (
            frame["satellite_code"].map(PLATFORM_NAMES).astype("string")
        )

        frame["sensor_family"] = metadata["sensor_family"]

        frame["source_row"] = pd.to_numeric(
            frame["source_row"], errors="raise"
        ).astype("Int64")

        frame["observation_id"] = [
            observation_id(product, source_file, source_row)
            for source_file, source_row in zip(
                frame["source_file"], frame["source_row"]
            )
        ]

        frame["latitude"] = pd.to_numeric(
            frame["latitude"], errors="coerce"
        ).astype("Float64")

        frame["longitude"] = pd.to_numeric(
            frame["longitude"], errors="coerce"
        ).astype("Float64")

        frame["acq_datetime_utc"], frame["acq_time_hhmm"], valid_time = (
            parse_utc_datetime(frame["acq_date"], frame["acq_time"])
        )

        if not valid_time.all():
            raise ValueError(
                f"{product}: invalid acquisition date/time rows "
                f"{int((~valid_time).sum())}"
            )

        frame["acq_date"] = pd.to_datetime(
            frame["acq_date"], errors="raise"
        ).dt.date

        frame["confidence_native"] = (
            frame["confidence"].astype("string").str.strip()
        )

        if metadata["sensor_family"] == "MODIS":
            frame["confidence_scheme"] = "MODIS_NUMERIC_0_100"

            frame["confidence_numeric"] = pd.to_numeric(
                frame["confidence_native"], errors="coerce"
            ).astype("Float64")

            frame["confidence_class"] = pd.Series(
                pd.NA, index=frame.index, dtype="string"
            )

            invalid_confidence = (
                frame["confidence_numeric"].isna()
                | ~frame["confidence_numeric"].between(0, 100)
            )

            if invalid_confidence.any():
                raise ValueError(
                    f"{product}: invalid MODIS numeric confidence"
                )

        else:
            frame["confidence_scheme"] = "VIIRS_CATEGORICAL_L_N_H"

            frame["confidence_class"] = (
                frame["confidence_native"].str.lower()
            )

            frame["confidence_numeric"] = pd.Series(
                pd.NA, index=frame.index, dtype="Float64"
            )

            invalid_confidence = ~frame["confidence_class"].isin(
                ["l", "n", "h"]
            )

            if invalid_confidence.any():
                raise ValueError(
                    f"{product}: unexpected VIIRS confidence classes: "
                    f"{sorted(frame.loc[invalid_confidence, 'confidence_class'].dropna().unique())}"
                )

        brightness = pd.to_numeric(
            frame["brightness"], errors="coerce"
        ).astype("Float64")

        bright_t31 = pd.to_numeric(
            frame["bright_t31"], errors="coerce"
        ).astype("Float64")

        frame["modis_brightness_21_22_k"] = pd.Series(
            pd.NA, index=frame.index, dtype="Float64"
        )
        frame["modis_brightness_t31_k"] = pd.Series(
            pd.NA, index=frame.index, dtype="Float64"
        )
        frame["viirs_brightness_i4_k"] = pd.Series(
            pd.NA, index=frame.index, dtype="Float64"
        )
        frame["viirs_brightness_i5_k"] = pd.Series(
            pd.NA, index=frame.index, dtype="Float64"
        )

        if metadata["sensor_family"] == "MODIS":
            frame["modis_brightness_21_22_k"] = brightness
            frame["modis_brightness_t31_k"] = bright_t31
        else:
            # Archive CSV field names: brightness and bright_t31.
            # Preserve their VIIRS I4/I5 meaning in the normalized schema.
            frame["viirs_brightness_i4_k"] = brightness
            frame["viirs_brightness_i5_k"] = bright_t31

        frame["frp_mw"] = pd.to_numeric(
            frame["frp"], errors="coerce"
        ).astype("Float64")

        frame["scan_native"] = pd.to_numeric(
            frame["scan"], errors="coerce"
        ).astype("Float64")

        frame["track_native"] = pd.to_numeric(
            frame["track"], errors="coerce"
        ).astype("Float64")

        frame["instrument_native"] = (
            frame["instrument"].astype("string")
        )

        frame["version_native"] = (
            frame["version"].astype("string")
        )

        frame["daynight"] = (
            frame["daynight"].astype("string")
        )

        frame["hotspot_type_native"] = (
            frame["type"].astype("string")
        )

        frame["source_product"] = (
            frame["source_product"].astype("string")
        )

        frame["source_file"] = (
            frame["source_file"].astype("string")
        )

        frame["archive_request_id"] = (
            frame["archive_request_id"].astype("string")
        )

        frame["source_archive"] = (
            frame["source_archive"].astype("string")
        )

        frame["source_archive_sha256"] = (
            frame["source_archive_sha256"].astype("string")
        )

        output_frames.append(frame)

    combined = pd.concat(output_frames, ignore_index=True)

    columns = [
        "observation_id",
        "source_product",
        "source_file",
        "source_row",
        "archive_request_id",
        "source_archive",
        "source_archive_sha256",
        "sensor_family",
        "satellite_code",
        "satellite_platform",
        "instrument_native",
        "latitude",
        "longitude",
        "acq_date",
        "acq_time_hhmm",
        "acq_datetime_utc",
        "confidence_native",
        "confidence_scheme",
        "confidence_numeric",
        "confidence_class",
        "modis_brightness_21_22_k",
        "modis_brightness_t31_k",
        "viirs_brightness_i4_k",
        "viirs_brightness_i5_k",
        "frp_mw",
        "scan_native",
        "track_native",
        "daynight",
        "version_native",
        "hotspot_type_native",
    ]

    combined = combined[columns].copy()

    if combined["observation_id"].duplicated().any():
        raise ValueError("Duplicate observation IDs detected")

    if combined["observation_id"].isna().any():
        raise ValueError("Null observation IDs detected")

    return combined


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Raw input not found: {INPUT}")

    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError(
            "Refusing to overwrite existing normalized output or report:\n"
            f"{OUTPUT}\n{REPORT}"
        )

    print("=" * 72)
    print("FIRESYNC | NASA FIRMS ARCHIVE NORMALIZATION")
    print("=" * 72)

    raw = pd.read_parquet(INPUT)

    print(f"Raw observations loaded: {len(raw):,}")

    normalized = normalize(raw)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    product_counts = (
        normalized.groupby("source_product")
        .size()
        .astype(int)
        .to_dict()
    )

    platform_counts = (
        normalized.groupby("satellite_platform")
        .size()
        .astype(int)
        .to_dict()
    )

    confidence_counts = (
        normalized.groupby("confidence_scheme")
        .size()
        .astype(int)
        .to_dict()
    )

    report = {
        "project": "FIRESYNC",
        "dataset": "NASA FIRMS archive normalized observations",
        "total_records": int(len(normalized)),
        "unique_observation_ids": int(
            normalized["observation_id"].nunique()
        ),
        "product_counts": product_counts,
        "platform_counts": platform_counts,
        "confidence_scheme_counts": confidence_counts,
        "invalid_utc_timestamps": int(
            normalized["acq_datetime_utc"].isna().sum()
        ),
        "brightness_mapping": {
            "MODIS": {
                "brightness": "modis_brightness_21_22_k",
                "bright_t31": "modis_brightness_t31_k",
            },
            "VIIRS": {
                "brightness": "viirs_brightness_i4_k",
                "bright_t31": "viirs_brightness_i5_k",
            },
        },
        "quality_note": (
            "Archive processing-quality classification has not been "
            "independently established. Native confidence schemes and "
            "sensor-specific brightness fields are preserved separately."
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(OUTPUT.relative_to(ROOT)),
    }

    temporary_output = OUTPUT.with_suffix(".tmp.parquet")
    temporary_report = REPORT.with_suffix(".tmp.json")

    try:
        normalized.to_parquet(
            temporary_output,
            index=False,
            engine="pyarrow",
        )

        with temporary_report.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

        temporary_output.replace(OUTPUT)
        temporary_report.replace(REPORT)

    except Exception:
        temporary_output.unlink(missing_ok=True)
        temporary_report.unlink(missing_ok=True)
        raise

    print("\nNORMALIZATION COMPLETE")
    print("-" * 72)
    print(f"Records: {len(normalized):,}")
    print(f"Unique IDs: {normalized['observation_id'].nunique():,}")
    print(f"Products: {product_counts}")
    print(f"Platforms: {platform_counts}")
    print(f"Confidence schemes: {confidence_counts}")
    print(f"\nParquet: {OUTPUT}")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
