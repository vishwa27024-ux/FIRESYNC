
from pathlib import Path

import pandas as pd
from pyproj import Geod


ROOT = Path(__file__).resolve().parents[1]

OBSERVATIONS_FILE = (
    ROOT / "data" / "processed"
    / "firms_archive_normalized_2021_2025.parquet"
)

BRIDGES_FILE = (
    ROOT / "data" / "interim"
    / "association_bridge_edge_details.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "interim"
    / "association_bridge_observation_inspection.csv"
)

GEOD = Geod(ellps="WGS84")


def main():
    observations = pd.read_parquet(OBSERVATIONS_FILE)
    bridges = pd.read_csv(BRIDGES_FILE)

    required = [
        "observation_id",
        "latitude",
        "longitude",
        "acq_datetime_utc",
        "sensor_family",
        "satellite_platform",
    ]

    missing = [
        column for column in required
        if column not in observations.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required observation columns: {missing}"
        )

    optional = [
        "confidence_native",
        "confidence_class",
        "confidence_numeric",
        "frp_mw",
        "scan_native",
        "track_native",
    ]

    detail_columns = required + [
        column for column in optional
        if column in observations.columns
    ]

    observations = observations[detail_columns].copy()
    observations["observation_id"] = (
        observations["observation_id"].astype(str)
    )

    bridges["modis_observation_id"] = (
        bridges["modis_observation_id"].astype(str)
    )
    bridges["viirs_observation_id"] = (
        bridges["viirs_observation_id"].astype(str)
    )

    modis = observations.rename(
        columns={
            column: f"modis_{column}"
            for column in detail_columns
            if column != "observation_id"
        }
    ).rename(
        columns={"observation_id": "modis_observation_id"}
    )

    viirs = observations.rename(
        columns={
            column: f"viirs_{column}"
            for column in detail_columns
            if column != "observation_id"
        }
    ).rename(
        columns={"observation_id": "viirs_observation_id"}
    )

    result = bridges.merge(
        modis,
        on="modis_observation_id",
        how="left",
        validate="many_to_one",
    ).merge(
        viirs,
        on="viirs_observation_id",
        how="left",
        validate="many_to_one",
    )

    if result["modis_latitude"].isna().any():
        raise ValueError("Some MODIS observations were not found.")

    if result["viirs_latitude"].isna().any():
        raise ValueError("Some VIIRS observations were not found.")

    result["modis_acq_datetime_utc"] = pd.to_datetime(
        result["modis_acq_datetime_utc"], utc=True
    )

    result["viirs_acq_datetime_utc"] = pd.to_datetime(
        result["viirs_acq_datetime_utc"], utc=True
    )

    distances = []
    signed_time_differences = []

    for _, row in result.iterrows():
        _, _, distance_m = GEOD.inv(
            row["modis_longitude"],
            row["modis_latitude"],
            row["viirs_longitude"],
            row["viirs_latitude"],
        )

        distances.append(distance_m / 1000)

        time_difference = (
            row["viirs_acq_datetime_utc"]
            - row["modis_acq_datetime_utc"]
        ).total_seconds() / 60

        signed_time_differences.append(time_difference)

    result["recalculated_distance_km"] = distances
    result["signed_time_difference_minutes"] = (
        signed_time_differences
    )

    result.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("FIRESYNC | BRIDGE OBSERVATION INSPECTION")
    print("=" * 70)

    print(f"Bridge edges inspected: {len(result)}")
    print(
        "Unmatched observation lookups: "
        f"{result[['modis_latitude', 'viirs_latitude']].isna().any(axis=1).sum()}"
    )

    display_columns = [
        "cluster_id",
        "modis_platform",
        "viirs_platform",
        "distance_km",
        "time_difference_minutes",
        "modis_latitude",
        "modis_longitude",
        "modis_acq_datetime_utc",
        "viirs_latitude",
        "viirs_longitude",
        "viirs_acq_datetime_utc",
    ]

    print("\nBRIDGE ENDPOINTS")
    print(
        result[display_columns].to_string(
            index=False
        )
    )

    print("\nCONFIDENCE AND FRP FIELDS")
    for column in [
        "modis_confidence_native",
        "viirs_confidence_native",
        "modis_confidence_class",
        "viirs_confidence_class",
        "modis_frp_mw",
        "viirs_frp_mw",
    ]:
        if column in result.columns:
            print(f"\n{column}")
            print(result[column].value_counts(dropna=False).to_string())

    print("\nOUTPUT:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
