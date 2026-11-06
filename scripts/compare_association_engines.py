
import json
import time
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from src.science.association_engine import build_associations
from src.science.optimized_association_engine import (
    build_associations_optimized,
)


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "march_2025_normalized_observations.parquet"
)

OUTPUT_DIR = ROOT / "data" / "interim"
OUTPUT_CSV = OUTPUT_DIR / "march_2025_engine_comparison.csv"
OUTPUT_JSON = OUTPUT_DIR / "march_2025_engine_comparison_report.json"

DISTANCE_THRESHOLDS = [0.5, 1.0, 2.0, 5.0]
TIME_THRESHOLDS = [30, 60, 90, 120]
REPEATS = 3


def compare_frames(reference, optimized, name):
    """Check that both engines produce equivalent DataFrames."""

    assert_frame_equal(
        reference.reset_index(drop=True),
        optimized.reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
        obj=name,
    )


def run_engine(engine, observations, distance, time_window):
    start = time.perf_counter()

    result = engine(
        observations,
        distance_threshold_km=distance,
        time_window_minutes=time_window,
    )

    elapsed = time.perf_counter() - start
    return result, elapsed


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Normalized dataset not found: {INPUT_FILE}"
        )

    observations = pd.read_parquet(INPUT_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("FIRESYNC | REFERENCE vs OPTIMIZED ASSOCIATION ENGINE")
    print("=" * 72)
    print(f"Observations: {len(observations)}")
    print(f"Benchmark repeats per configuration: {REPEATS}")
    print()

    report_rows = []
    total_configurations = (
        len(DISTANCE_THRESHOLDS) * len(TIME_THRESHOLDS)
    )

    configuration = 0

    for time_window in TIME_THRESHOLDS:
        for distance in DISTANCE_THRESHOLDS:
            configuration += 1

            reference_times = []
            optimized_times = []

            reference_result = None
            optimized_result = None

            for _ in range(REPEATS):
                reference_result, elapsed = run_engine(
                    build_associations,
                    observations,
                    distance,
                    time_window,
                )
                reference_times.append(elapsed)

                optimized_result, elapsed = run_engine(
                    build_associations_optimized,
                    observations,
                    distance,
                    time_window,
                )
                optimized_times.append(elapsed)

            # Compare all three output DataFrames.
            compare_frames(
                reference_result.edges,
                optimized_result.edges,
                "candidate edges",
            )

            compare_frames(
                reference_result.clusters,
                optimized_result.clusters,
                "association clusters",
            )

            compare_frames(
                reference_result.unmatched,
                optimized_result.unmatched,
                "unmatched observations",
            )

            reference_median = (
                sorted(reference_times)[len(reference_times) // 2]
            )
            optimized_median = (
                sorted(optimized_times)[len(optimized_times) // 2]
            )

            speedup = (
                reference_median / optimized_median
                if optimized_median > 0
                else None
            )

            row = {
                "distance_threshold_km": distance,
                "time_window_minutes": time_window,
                "candidate_edge_count": len(reference_result.edges),
                "cluster_count": len(reference_result.clusters),
                "unmatched_observation_count": len(
                    reference_result.unmatched
                ),
                "reference_median_seconds": reference_median,
                "optimized_median_seconds": optimized_median,
                "measured_speedup": speedup,
                "outputs_identical": True,
            }

            report_rows.append(row)

            print(
                f"[{configuration:02d}/{total_configurations}] "
                f"{distance:>3} km | {time_window:>3} min | "
                f"edges={row['candidate_edge_count']} | "
                f"clusters={row['cluster_count']} | "
                f"unmatched={row['unmatched_observation_count']} | "
                f"verified"
            )

    report = pd.DataFrame(report_rows)
    report.to_csv(OUTPUT_CSV, index=False)

    summary = {
        "dataset": str(INPUT_FILE.relative_to(ROOT)),
        "observation_count": len(observations),
        "configuration_count": len(report),
        "benchmark_repeats_per_engine": REPEATS,
        "all_configurations_verified": bool(
            report["outputs_identical"].all()
        ),
        "all_candidate_edge_counts": sorted(
            report["candidate_edge_count"].unique().tolist()
        ),
        "all_cluster_counts": sorted(
            report["cluster_count"].unique().tolist()
        ),
        "all_unmatched_counts": sorted(
            report["unmatched_observation_count"].unique().tolist()
        ),
        "reference_median_seconds_over_configurations": float(
            report["reference_median_seconds"].median()
        ),
        "optimized_median_seconds_over_configurations": float(
            report["optimized_median_seconds"].median()
        ),
        "median_measured_speedup": float(
            report["measured_speedup"].median()
        ),
        "benchmark_warning": (
            "The March sample is small. Timing results are preliminary "
            "and do not establish scalability or production speedup."
        ),
    }

    OUTPUT_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("COMPARISON COMPLETE")
    print("=" * 72)
    print(f"All configurations verified: {summary['all_configurations_verified']}")
    print(f"CSV report: {OUTPUT_CSV}")
    print(f"JSON report: {OUTPUT_JSON}")
    print()
    print("Median reference time: "
          f"{summary['reference_median_seconds_over_configurations']:.6f}s")
    print("Median optimized time: "
          f"{summary['optimized_median_seconds_over_configurations']:.6f}s")
    print("Median measured speedup: "
          f"{summary['median_measured_speedup']:.2f}x")
    print()
    print("Note: This is a small-sample benchmark, not a scalability claim.")


if __name__ == "__main__":
    main()
