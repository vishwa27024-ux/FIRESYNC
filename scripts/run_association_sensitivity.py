
"""Run FIRESYNC association threshold sensitivity analysis."""

import json
from pathlib import Path

import pandas as pd

from src.science.association_engine import build_associations


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "march_2025_normalized_observations.parquet"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "association"
REPORT_DIR = ROOT / "data" / "interim"

DISTANCE_THRESHOLDS_KM = [0.5, 1.0, 2.0, 5.0]
TIME_THRESHOLDS_MINUTES = [30, 60, 90, 120]


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    observations = pd.read_parquet(INPUT_FILE)

    sensitivity_records = []
    default_result = None

    print("=" * 65)
    print("FIRESYNC | ASSOCIATION SENSITIVITY ANALYSIS")
    print("=" * 65)
    print(f"Input observations: {len(observations)}")

    for distance_km in DISTANCE_THRESHOLDS_KM:
        for time_minutes in TIME_THRESHOLDS_MINUTES:

            result = build_associations(
                observations,
                distance_threshold_km=distance_km,
                time_window_minutes=time_minutes,
            )

            ambiguity_counts = (
                result.clusters["ambiguity_type"]
                .value_counts()
                .to_dict()
            )

            sensitivity_records.append({
                "distance_threshold_km": distance_km,
                "time_window_minutes": time_minutes,
                "candidate_edge_count": len(result.edges),
                "cluster_count": len(result.clusters),
                "unmatched_observation_count": len(result.unmatched),
                "one_to_one_cluster_count": ambiguity_counts.get(
                    "one_to_one_candidate", 0
                ),
                "one_to_many_cluster_count": ambiguity_counts.get(
                    "one_to_many", 0
                ),
                "many_to_one_cluster_count": ambiguity_counts.get(
                    "many_to_one", 0
                ),
                "many_to_many_cluster_count": ambiguity_counts.get(
                    "many_to_many", 0
                ),
            })

            print(
                f"{distance_km:>4.1f} km | "
                f"{time_minutes:>3} min | "
                f"edges={len(result.edges):>3} | "
                f"clusters={len(result.clusters):>3} | "
                f"unmatched={len(result.unmatched):>3}"
            )

            if distance_km == 1.0 and time_minutes == 60:
                default_result = result

    sensitivity = pd.DataFrame(sensitivity_records)

    sensitivity_path = (
        REPORT_DIR / "march_2025_association_sensitivity.csv"
    )
    sensitivity.to_csv(sensitivity_path, index=False)

    if default_result is None:
        raise RuntimeError("Default 1 km / 60 minute result was not generated.")

    # Save the default association outputs.
    default_result.edges.to_csv(
        OUTPUT_DIR / "march_2025_candidate_edges.csv",
        index=False,
    )

    default_result.clusters.to_csv(
        OUTPUT_DIR / "march_2025_association_clusters.csv",
        index=False,
    )

    default_result.unmatched.to_parquet(
        OUTPUT_DIR / "march_2025_unmatched_observations.parquet",
        index=False,
    )

    # Save a compact machine-readable summary.
    report = {
        "project": "FIRESYNC",
        "input_file": str(INPUT_FILE.relative_to(ROOT)),
        "input_observation_count": len(observations),
        "analysis": "MODIS-VIIRS candidate association",
        "association_is_confirmed_fire": False,
        "thresholds": {
            "distance_km": DISTANCE_THRESHOLDS_KM,
            "time_minutes": TIME_THRESHOLDS_MINUTES,
        },
        "default_configuration": {
            "distance_km": 1.0,
            "time_minutes": 60,
            "candidate_edge_count": len(default_result.edges),
            "cluster_count": len(default_result.clusters),
            "unmatched_observation_count": len(
                default_result.unmatched
            ),
        },
        "sensitivity_csv": str(sensitivity_path.relative_to(ROOT)),
    }

    report_path = REPORT_DIR / "march_2025_association_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("ANALYSIS COMPLETE")
    print("=" * 65)
    print(f"Sensitivity CSV: {sensitivity_path}")
    print(f"Candidate edges: {OUTPUT_DIR / 'march_2025_candidate_edges.csv'}")
    print(f"Clusters: {OUTPUT_DIR / 'march_2025_association_clusters.csv'}")
    print(f"Unmatched: {OUTPUT_DIR / 'march_2025_unmatched_observations.parquet'}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
