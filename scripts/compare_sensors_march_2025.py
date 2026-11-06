
from pathlib import Path
from datetime import datetime, timezone
import json

import pandas as pd
from pyproj import Geod


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data" / "processed"
    / "march_2025_normalized_observations.parquet"
)

OUTPUT_DIR = ROOT / "data" / "interim"

PAIRWISE_FILE = (
    OUTPUT_DIR / "march_2025_modis_viirs_pairwise.csv"
)

CANDIDATE_FILE = (
    OUTPUT_DIR / "march_2025_candidate_pairs.csv"
)

SENSITIVITY_FILE = (
    OUTPUT_DIR / "march_2025_matching_sensitivity.csv"
)

REPORT_FILE = (
    OUTPUT_DIR / "march_2025_sensor_comparison_report.json"
)

TIME_THRESHOLDS_MIN = [30, 60, 90, 120]
DISTANCE_THRESHOLDS_KM = [0.5, 1.0, 2.0, 5.0]

GEOD = Geod(ellps="WGS84")


def main():
    print("=" * 65)
    print("FIRESYNC | MODIS-VIIRS PAIRWISE COMPARISON")
    print("=" * 65)

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    outputs = [
        PAIRWISE_FILE,
        CANDIDATE_FILE,
        SENSITIVITY_FILE,
        REPORT_FILE,
    ]

    for output in outputs:
        if output.exists():
            raise FileExistsError(
                f"Output already exists; refusing overwrite: {output}"
            )

    data = pd.read_parquet(INPUT)

    required = {
        "observation_id",
        "sensor_family",
        "source_product",
        "satellite_platform",
        "latitude",
        "longitude",
        "acq_datetime_utc",
        "confidence_native",
        "frp_mw",
    }

    missing = required - set(data.columns)

    if missing:
        raise RuntimeError(
            f"Normalized dataset missing fields: {sorted(missing)}"
        )

    data["acq_datetime_utc"] = pd.to_datetime(
        data["acq_datetime_utc"],
        utc=True,
        errors="coerce",
    )

    if data["acq_datetime_utc"].isna().any():
        raise RuntimeError("Invalid UTC observation timestamps.")

    modis = data[
        data["sensor_family"] == "MODIS"
    ].copy()

    viirs = data[
        data["sensor_family"] == "VIIRS"
    ].copy()

    if modis.empty or viirs.empty:
        raise RuntimeError(
            "Both MODIS and VIIRS observations are required."
        )

    print(f"\nMODIS observations: {len(modis)}")
    print(f"VIIRS observations: {len(viirs)}")

    # Every pair is retained for transparent sensitivity analysis.
    # No one-to-one matching or fire-event identity is inferred.
    pairs = []

    for _, m in modis.iterrows():
        for _, v in viirs.iterrows():

            _, _, distance_m = GEOD.inv(
                float(m["longitude"]),
                float(m["latitude"]),
                float(v["longitude"]),
                float(v["latitude"]),
            )

            time_difference = abs(
                (
                    m["acq_datetime_utc"]
                    - v["acq_datetime_utc"]
                ).total_seconds()
            ) / 60.0

            pairs.append({
                "modis_observation_id": m["observation_id"],
                "viirs_observation_id": v["observation_id"],
                "modis_platform": m["satellite_platform"],
                "viirs_platform": v["satellite_platform"],
                "modis_product": m["source_product"],
                "viirs_product": v["source_product"],
                "modis_datetime_utc": m["acq_datetime_utc"],
                "viirs_datetime_utc": v["acq_datetime_utc"],
                "modis_latitude": m["latitude"],
                "modis_longitude": m["longitude"],
                "viirs_latitude": v["latitude"],
                "viirs_longitude": v["longitude"],
                "spatial_distance_km": distance_m / 1000.0,
                "time_difference_minutes": time_difference,
                "modis_confidence_native": str(
                    m["confidence_native"]
                ),
                "viirs_confidence_native": str(
                    v["confidence_native"]
                ),
                "modis_frp_mw": m["frp_mw"],
                "viirs_frp_mw": v["frp_mw"],
            })

    pairwise = pd.DataFrame(pairs)

    expected_pairs = len(modis) * len(viirs)

    if len(pairwise) != expected_pairs:
        raise RuntimeError("Unexpected pairwise record count.")

    print(f"\nPairwise combinations: {len(pairwise)}")

    # Maximum exploratory envelope. It is not a final match definition.
    candidates = pairwise[
        pairwise["spatial_distance_km"] <= 5.0
    ].copy()

    candidates = candidates[
        candidates["time_difference_minutes"] <= 120
    ].copy()

    # Evaluate all temporal and spatial threshold combinations.
    sensitivity = []

    for time_limit in TIME_THRESHOLDS_MIN:
        for distance_limit in DISTANCE_THRESHOLDS_KM:

            selected = pairwise[
                (
                    pairwise["time_difference_minutes"]
                    <= time_limit
                )
                & (
                    pairwise["spatial_distance_km"]
                    <= distance_limit
                )
            ]

            sensitivity.append({
                "time_threshold_minutes": time_limit,
                "distance_threshold_km": distance_limit,
                "candidate_pair_count": len(selected),
                "unique_modis_observations": (
                    selected["modis_observation_id"].nunique()
                ),
                "unique_viirs_observations": (
                    selected["viirs_observation_id"].nunique()
                ),
            })

    sensitivity_data = pd.DataFrame(sensitivity)

    # Nearest VIIRS observation for each MODIS observation.
    nearest_rows = []

    for observation_id, group in pairwise.groupby(
        "modis_observation_id"
    ):
        nearest = group.sort_values(
            "spatial_distance_km"
        ).iloc[0]

        nearest_rows.append({
            "modis_observation_id": observation_id,
            "nearest_viirs_observation_id": (
                nearest["viirs_observation_id"]
            ),
            "nearest_viirs_platform": (
                nearest["viirs_platform"]
            ),
            "nearest_spatial_distance_km": (
                nearest["spatial_distance_km"]
            ),
            "time_difference_minutes": (
                nearest["time_difference_minutes"]
            ),
        })

    nearest_data = pd.DataFrame(nearest_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairwise.to_csv(PAIRWISE_FILE, index=False)
    candidates.to_csv(CANDIDATE_FILE, index=False)
    sensitivity_data.to_csv(SENSITIVITY_FILE, index=False)

    report = {
        "project": "FIRESYNC",
        "period": "2025-03-01 to 2025-03-31",
        "region": (
            "Bandipur landscape acquisition rectangle; "
            "not an official protected-area boundary."
        ),
        "method": "Exhaustive pairwise MODIS-VIIRS distance and time comparison",
        "distance_method": "WGS84 geodesic inverse calculation",
        "time_method": "Absolute difference between UTC timestamps",
        "modis_observations": len(modis),
        "viirs_observations": len(viirs),
        "pairwise_combinations": len(pairwise),
        "candidate_pairs_within_5km_and_120min": len(candidates),
        "time_thresholds_minutes": TIME_THRESHOLDS_MIN,
        "distance_thresholds_km": DISTANCE_THRESHOLDS_KM,
        "sensitivity_results": sensitivity,
        "nearest_viirs_per_modis": nearest_rows,
        "matching_interpretation": (
            "Candidate spatial-temporal association only. "
            "Not confirmation of a shared fire event."
        ),
        "pixel_footprints_inferred": False,
        "one_to_one_matching_enforced": False,
        "confidence_values_compared_across_sensors": False,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "outputs": [
            PAIRWISE_FILE.name,
            CANDIDATE_FILE.name,
            SENSITIVITY_FILE.name,
        ],
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nSENSITIVITY ANALYSIS")
    print(sensitivity_data.to_string(index=False))

    print("\nNEAREST VIIRS OBSERVATION FOR EACH MODIS RECORD")
    print(nearest_data.to_string(index=False))

    print("\nMAXIMUM EXPLORATORY ENVELOPE")
    print("Spatial threshold: 5 km")
    print("Temporal threshold: 120 minutes")
    print("Candidate pairs:", len(candidates))

    print("\nOUTPUTS")
    print(PAIRWISE_FILE)
    print(CANDIDATE_FILE)
    print(SENSITIVITY_FILE)
    print(REPORT_FILE)

    print("\nNo observations were merged or deleted.")
    print("No fire-event identity has been inferred.")
    print("=" * 65)


if __name__ == "__main__":
    main()
