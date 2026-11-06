
from pathlib import Path

import pandas as pd
from pyproj import Geod


ROOT = Path(__file__).resolve().parents[1]

CANDIDATES = (
    ROOT / "data" / "interim"
    / "march_2025_candidate_pairs.csv"
)

DATASET = (
    ROOT / "data" / "processed"
    / "march_2025_normalized_observations.parquet"
)

GEOD = Geod(ellps="WGS84")


def distance_km(a, b):
    _, _, metres = GEOD.inv(
        float(a["longitude"]),
        float(a["latitude"]),
        float(b["longitude"]),
        float(b["latitude"]),
    )
    return metres / 1000


def main():
    pairs = pd.read_csv(CANDIDATES)
    data = pd.read_parquet(DATASET)

    modis_ids = set(pairs["modis_observation_id"])
    viirs_ids = set(pairs["viirs_observation_id"])

    ids = modis_ids | viirs_ids

    selected = data[
        data["observation_id"].isin(ids)
    ].copy()

    if len(selected) != len(ids):
        raise RuntimeError(
            "Could not retrieve every unique observation."
        )

    selected = selected.sort_values(
        ["sensor_family", "latitude", "longitude"]
    )

    print("=" * 70)
    print("FIRESYNC | CANDIDATE CLUSTER INVESTIGATION")
    print("=" * 70)

    fields = [
        "observation_id",
        "source_product",
        "satellite_platform",
        "acq_datetime_utc",
        "latitude",
        "longitude",
        "confidence_native",
        "confidence_scheme",
        "frp_mw",
        "scan_native",
        "track_native",
    ]

    print("\nFOUR UNDERLYING OBSERVATIONS\n")
    print(selected[fields].to_string(index=False))

    print("\nPAIRWISE DISTANCE MATRIX (KM)\n")

    matrix = pairs.pivot(
        index="modis_observation_id",
        columns="viirs_observation_id",
        values="spatial_distance_km",
    )

    print(matrix.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nPAIRWISE TIME DIFFERENCES (MINUTES)\n")

    time_matrix = pairs.pivot(
        index="modis_observation_id",
        columns="viirs_observation_id",
        values="time_difference_minutes",
    )

    print(time_matrix.to_string())

    print("\nWITHIN-SENSOR SEPARATIONS\n")

    for sensor in ["MODIS", "VIIRS"]:
        group = selected[
            selected["sensor_family"] == sensor
        ]

        print(f"\n{sensor}")

        if len(group) < 2:
            print("Not enough observations.")
            continue

        rows = list(group.to_dict(orient="records"))

        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                spatial = distance_km(rows[i], rows[j])

                temporal = abs(
                    (
                        pd.Timestamp(rows[i]["acq_datetime_utc"])
                        - pd.Timestamp(rows[j]["acq_datetime_utc"])
                    ).total_seconds()
                ) / 60

                print(
                    f"Distance: {spatial:.4f} km | "
                    f"Time difference: {temporal:.2f} minutes"
                )

    print("\nUNIQUE OBSERVATIONS:", len(selected))
    print("CANDIDATE RELATIONSHIPS:", len(pairs))
    print("No observations were modified.")
    print("=" * 70)


if __name__ == "__main__":
    main()
