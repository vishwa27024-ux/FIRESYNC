
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data" / "processed"
    / "march_2025_landscape_observations.parquet"
)

OUTPUT_DIR = ROOT / "data" / "processed"

OUTPUT = (
    OUTPUT_DIR / "march_2025_normalized_observations.parquet"
)

REPORT = (
    ROOT / "data" / "interim"
    / "march_2025_normalization_report.json"
)

PRODUCTS = {
    "MODIS_SP": "MODIS",
    "VIIRS_SNPP_SP": "VIIRS",
    "VIIRS_NOAA20_SP": "VIIRS",
    "VIIRS_NOAA21_SP": "VIIRS",
}


PLATFORM_NAMES = {
    "T": "Terra",
    "TERRA": "Terra",
    "A": "Aqua",
    "AQUA": "Aqua",
    "N": "Suomi-NPP",
    "N20": "NOAA-20",
    "N21": "NOAA-21",
}


VIIRS_CONFIDENCE = {
    "l": "low",
    "n": "nominal",
    "h": "high",
}


def numeric_column(data, column):
    """Return a nullable numeric series for an optional field."""
    if column not in data.columns:
        return pd.Series(
            pd.array([pd.NA] * len(data), dtype="Float64"),
            index=data.index,
        )

    return pd.to_numeric(
        data[column], errors="coerce"
    ).astype("Float64")


def make_observation_id(row):
    """Create a reproducible ID tied to source and original row."""
    identity = (
        f"{row['source_product']}|"
        f"{row['source_file']}|"
        f"{row['source_row']}"
    )

    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()


def main():
    print("=" * 65)
    print("FIRESYNC | SENSOR-AWARE NORMALIZATION")
    print("=" * 65)

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError(
            "Normalized output already exists. "
            "Refusing to overwrite."
        )

    raw = pd.read_parquet(INPUT)

    if raw.empty:
        raise RuntimeError("Input dataset is empty.")

    required = {
        "source_product",
        "source_file",
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
        "frp",
        "daynight",
    }

    missing = required - set(raw.columns)

    if missing:
        raise RuntimeError(
            f"Input dataset missing columns: {sorted(missing)}"
        )

    raw = raw.copy()

    if not set(raw["source_product"].unique()).issubset(PRODUCTS):
        raise RuntimeError("Unknown source product found.")

    raw["sensor_family"] = raw["source_product"].map(PRODUCTS)

    # Preserve the original observation order within each source file.
    raw["source_row"] = (
        raw.groupby(
            ["source_product", "source_file"],
            sort=False,
        ).cumcount()
    )

    result = pd.DataFrame(index=raw.index)

    # Source identity and reproducible provenance.
    result["observation_id"] = raw.apply(
        make_observation_id,
        axis=1,
    )

    result["source_product"] = raw["source_product"].astype("string")
    result["source_file"] = raw["source_file"].astype("string")
    result["source_row"] = raw["source_row"].astype("Int64")

    result["sensor_family"] = raw["sensor_family"].astype("string")

    # Satellite platform.
    satellite_code = (
        raw["satellite"].astype("string").str.strip().str.upper()
    )

    unknown_platforms = set(
        satellite_code.dropna().unique()
    ) - set(PLATFORM_NAMES)

    if unknown_platforms:
        raise RuntimeError(
            f"Unknown satellite identifiers: {unknown_platforms}"
        )

    result["satellite_code"] = satellite_code
    result["satellite_platform"] = (
        satellite_code.map(PLATFORM_NAMES).astype("string")
    )

    # Instrument.
    result["instrument_native"] = (
        raw["instrument"].astype("string")
    )

    # Coordinates.
    result["latitude"] = numeric_column(raw, "latitude")
    result["longitude"] = numeric_column(raw, "longitude")

    if result[["latitude", "longitude"]].isna().any().any():
        raise RuntimeError("Missing or invalid coordinates.")

    if not result["latitude"].between(-90, 90).all():
        raise RuntimeError("Latitude outside valid range.")

    if not result["longitude"].between(-180, 180).all():
        raise RuntimeError("Longitude outside valid range.")

    # Acquisition time: NASA FIRMS HHMM is UTC.
    date_text = raw["acq_date"].astype("string").str.strip()
    time_numeric = pd.to_numeric(
        raw["acq_time"], errors="coerce"
    )

    if time_numeric.isna().any():
        raise RuntimeError("Invalid acquisition time values.")

    if not (time_numeric % 1 == 0).all():
        raise RuntimeError("Acquisition times must be integer HHMM.")

    time_integer = time_numeric.astype("Int64")

    hours = time_integer // 100
    minutes = time_integer % 100

    if (
        (hours > 23).any()
        or (minutes > 59).any()
    ):
        raise RuntimeError("Acquisition time outside UTC clock range.")

    time_hhmm = time_integer.astype("string").str.zfill(4)

    timestamp = pd.to_datetime(
        date_text + " " + time_hhmm,
        format="%Y-%m-%d %H%M",
        utc=True,
        errors="coerce",
    )

    if timestamp.isna().any():
        raise RuntimeError("Failed to create UTC timestamps.")

    result["acq_date"] = pd.to_datetime(
        date_text, format="%Y-%m-%d", errors="raise"
    ).dt.date

    result["acq_time_hhmm"] = time_hhmm
    result["acq_datetime_utc"] = timestamp

    # Confidence: keep the native representation and scheme.
    confidence = (
        raw["confidence"].astype("string").str.strip()
    )

    result["confidence_native"] = confidence
    result["confidence_scheme"] = raw["sensor_family"].map({
        "MODIS": "MODIS_NUMERIC_0_100",
        "VIIRS": "VIIRS_CATEGORICAL_L_N_H",
    }).astype("string")

    modis_mask = raw["sensor_family"] == "MODIS"
    viirs_mask = raw["sensor_family"] == "VIIRS"

    confidence_numeric = pd.Series(
        pd.array([pd.NA] * len(raw), dtype="Float64"),
        index=raw.index,
    )

    confidence_class = pd.Series(
        pd.array([pd.NA] * len(raw), dtype="string"),
        index=raw.index,
    )

    modis_values = pd.to_numeric(
        confidence.loc[modis_mask],
        errors="coerce",
    )

    if modis_values.isna().any():
        raise RuntimeError("Invalid MODIS confidence value.")

    if not modis_values.between(0, 100).all():
        raise RuntimeError("MODIS confidence outside 0-100.")

    confidence_numeric.loc[modis_mask] = modis_values

    viirs_values = (
        confidence.loc[viirs_mask].str.lower()
    )

    unknown_confidence = set(
        viirs_values.dropna().unique()
    ) - set(VIIRS_CONFIDENCE)

    if unknown_confidence:
        raise RuntimeError(
            f"Unknown VIIRS confidence classes: {unknown_confidence}"
        )

    if viirs_values.isna().any():
        raise RuntimeError("Missing VIIRS confidence class.")

    confidence_class.loc[viirs_mask] = (
        viirs_values.map(VIIRS_CONFIDENCE)
    )

    result["confidence_numeric"] = confidence_numeric
    result["confidence_class"] = confidence_class

    # Brightness: retain individual sensor/channel measurements.
    result["modis_brightness_21_22_k"] = numeric_column(
        raw, "brightness"
    ).where(modis_mask)

    result["modis_brightness_t31_k"] = numeric_column(
        raw, "bright_t31"
    ).where(modis_mask)

    viirs_brightness_source = (
        raw["bright_ti4"]
        if "bright_ti4" in raw.columns
        else raw.get("brightness")
    )

    if viirs_brightness_source is None:
        viirs_brightness = pd.Series(
            pd.array([pd.NA] * len(raw), dtype="Float64"),
            index=raw.index,
        )
    else:
        viirs_brightness = pd.to_numeric(
            viirs_brightness_source, errors="coerce"
        ).astype("Float64")

    result["viirs_brightness_i4_k"] = (
        viirs_brightness.where(viirs_mask)
    )

    result["viirs_brightness_i5_k"] = (
        numeric_column(raw, "bright_ti5").where(viirs_mask)
    )

    # FRP has a common physical unit, but product-specific
    # retrieval differences must still be considered.
    result["frp_mw"] = numeric_column(raw, "frp")

    # Preserve reported scan/track values; no footprint geometry
    # or pixel-area estimate is inferred here.
    result["scan_native"] = numeric_column(raw, "scan")
    result["track_native"] = numeric_column(raw, "track")

    result["daynight"] = (
        raw["daynight"].astype("string").str.upper()
    )

    result["version_native"] = (
        raw["version"].astype("string")
    )

    if "type" in raw.columns:
        result["hotspot_type_native"] = (
            raw["type"].astype("string")
        )

    if result["observation_id"].duplicated().any():
        raise RuntimeError("Observation IDs are not unique.")

    # Keep source-product and date coverage transparent.
    result = result.sort_values(
        ["acq_datetime_utc", "source_product", "observation_id"]
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result.to_parquet(
        OUTPUT,
        index=False,
    )

    product_counts = (
        result.groupby("source_product")
        .size()
        .to_dict()
    )

    confidence_summary = (
        result.groupby(
            ["source_product", "confidence_native"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .to_dict(orient="records")
    )

    report = {
        "project": "FIRESYNC",
        "dataset": OUTPUT.name,
        "input_dataset": INPUT.name,
        "row_count": len(result),
        "unique_observation_ids": int(
            result["observation_id"].nunique()
        ),
        "product_counts": {
            key: int(value)
            for key, value in product_counts.items()
        },
        "confidence_summary": confidence_summary,
        "confidence_handling": (
            "Native confidence preserved. MODIS numeric and "
            "VIIRS categorical confidence are kept separate."
        ),
        "brightness_handling": (
            "MODIS and VIIRS channel measurements are stored "
            "in separate columns."
        ),
        "time_handling": (
            "Acquisition time parsed as UTC using HHMM values."
        ),
        "scan_track_handling": (
            "Native values retained. No pixel footprint inferred."
        ),
        "source_data_modified": False,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)

    REPORT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nNORMALIZATION COMPLETE")
    print("-" * 65)
    print("Rows:", len(result))
    print("Unique observation IDs:", result["observation_id"].nunique())

    print("\nPRODUCT COUNTS")
    print(result.groupby("source_product").size().to_string())

    print("\nCONFIDENCE SCHEMES")
    print(
        result.groupby(
            ["source_product", "confidence_scheme"]
        ).size().to_string()
    )

    print("\nOUTPUTS")
    print("Normalized Parquet:", OUTPUT)
    print("Normalization report:", REPORT)

    print("\nOriginal raw and combined datasets unchanged.")
    print("=" * 65)


if __name__ == "__main__":
    main()
