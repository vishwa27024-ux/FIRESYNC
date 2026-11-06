
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.science.association_engine import build_associations
from src.science.optimized_association_engine import (
    build_associations_optimized,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "interim"

OUTPUT_CSV = OUTPUT_DIR / "association_scalability_benchmark.csv"
OUTPUT_JSON = OUTPUT_DIR / "association_scalability_benchmark.json"

RANDOM_SEED = 42

MODIS_COUNT = 1_000
VIIRS_COUNT = 3_000
PLANTED_PAIRS = 500

DISTANCE_KM = 1.0
TIME_MINUTES = 60.0
REPEATS = 3

# Bandipur landscape acquisition rectangle.
WEST, SOUTH = 76.15, 11.55
EAST, NORTH = 76.91, 11.96


def generate_synthetic_observations():
    rng = np.random.default_rng(RANDOM_SEED)

    modis_lat = rng.uniform(SOUTH, NORTH, MODIS_COUNT)
    modis_lon = rng.uniform(WEST, EAST, MODIS_COUNT)

    # Random MODIS acquisition times within a 24-hour period.
    base_time = pd.Timestamp("2025-03-15T00:00:00Z")

    modis_seconds = rng.integers(
        0, 24 * 60 * 60, size=MODIS_COUNT
    )

    modis_times = (
        base_time
        + pd.to_timedelta(modis_seconds, unit="s")
    )

    # Generate ordinary VIIRS observations across the same area.
    viirs_lat = rng.uniform(SOUTH, NORTH, VIIRS_COUNT)
    viirs_lon = rng.uniform(WEST, EAST, VIIRS_COUNT)

    viirs_seconds = rng.integers(
        0, 24 * 60 * 60, size=VIIRS_COUNT
    )

    viirs_times = (
        base_time
        + pd.to_timedelta(viirs_seconds, unit="s")
    )

    # Plant controlled nearby, time-compatible observations.
    # These are synthetic candidate pairs, not real fire matches.
    viirs_lat[:PLANTED_PAIRS] = (
        modis_lat[:PLANTED_PAIRS]
        + rng.normal(0, 0.0005, PLANTED_PAIRS)
    )

    viirs_lon[:PLANTED_PAIRS] = (
        modis_lon[:PLANTED_PAIRS]
        + rng.normal(0, 0.0005, PLANTED_PAIRS)
    )

    viirs_times = viirs_times.to_series().reset_index(drop=True)

    viirs_times.iloc[:PLANTED_PAIRS] = (
    pd.Series(modis_times[:PLANTED_PAIRS])
    + pd.Timedelta(minutes=17)
).to_numpy()

    viirs_times = pd.DatetimeIndex(viirs_times)

    modis = pd.DataFrame({
        "observation_id": [
            f"SYN-M-{i:05d}" for i in range(MODIS_COUNT)
        ],
        "sensor_family": "MODIS",
        "satellite_platform": "Aqua",
        "latitude": modis_lat,
        "longitude": modis_lon,
        "acq_datetime_utc": modis_times,
    })

    viirs = pd.DataFrame({
        "observation_id": [
            f"SYN-V-{i:05d}" for i in range(VIIRS_COUNT)
        ],
        "sensor_family": "VIIRS",
        "satellite_platform": "NOAA-20",
        "latitude": viirs_lat,
        "longitude": viirs_lon,
        "acq_datetime_utc": viirs_times,
    })

    observations = pd.concat(
        [modis, viirs],
        ignore_index=True,
    )

    return observations


def verify_outputs(reference, optimized):
    assert_frame_equal(
        reference.edges.reset_index(drop=True),
        optimized.edges.reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
        obj="candidate edges",
    )

    assert_frame_equal(
        reference.clusters.reset_index(drop=True),
        optimized.clusters.reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
        obj="association clusters",
    )

    assert_frame_equal(
        reference.unmatched.reset_index(drop=True),
        optimized.unmatched.reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
        obj="unmatched observations",
    )


def benchmark_engine(engine, observations):
    durations = []
    result = None

    for _ in range(REPEATS):
        start = time.perf_counter()

        result = engine(
            observations,
            distance_threshold_km=DISTANCE_KM,
            time_window_minutes=TIME_MINUTES,
        )

        durations.append(time.perf_counter() - start)

    return result, durations


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("FIRESYNC | ASSOCIATION ENGINE SCALABILITY BENCHMARK")
    print("=" * 72)

    observations = generate_synthetic_observations()

    print(f"Total observations: {len(observations)}")
    print(f"MODIS observations: {MODIS_COUNT}")
    print(f"VIIRS observations: {VIIRS_COUNT}")
    print(f"Planted nearby pairs: {PLANTED_PAIRS}")
    print(f"Spatial threshold: {DISTANCE_KM} km")
    print(f"Temporal threshold: {TIME_MINUTES} minutes")
    print(f"Benchmark repetitions: {REPEATS}")
    print()

    print("Running reference engine...")
    reference_result, reference_times = benchmark_engine(
        build_associations,
        observations,
    )

    print("Running optimized engine...")
    optimized_result, optimized_times = benchmark_engine(
        build_associations_optimized,
        observations,
    )

    print("Verifying output equivalence...")

    verify_outputs(reference_result, optimized_result)

    reference_median = float(np.median(reference_times))
    optimized_median = float(np.median(optimized_times))

    speedup = (
        reference_median / optimized_median
        if optimized_median > 0
        else None
    )

    report = {
        "random_seed": RANDOM_SEED,
        "total_observations": len(observations),
        "modis_observations": MODIS_COUNT,
        "viirs_observations": VIIRS_COUNT,
        "planted_nearby_pairs": PLANTED_PAIRS,
        "distance_threshold_km": DISTANCE_KM,
        "time_window_minutes": TIME_MINUTES,
        "benchmark_repeats": REPEATS,
        "candidate_edge_count": len(reference_result.edges),
        "cluster_count": len(reference_result.clusters),
        "unmatched_observation_count": len(
            reference_result.unmatched
        ),
        "outputs_identical": True,
        "reference_times_seconds": reference_times,
        "optimized_times_seconds": optimized_times,
        "reference_median_seconds": reference_median,
        "optimized_median_seconds": optimized_median,
        "measured_speedup": speedup,
        "benchmark_note": (
            "Synthetic benchmark only. Results do not establish "
            "scientific validity or production performance."
        ),
    }

    pd.DataFrame([report]).to_csv(
        OUTPUT_CSV,
        index=False,
    )

    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("BENCHMARK RESULTS")
    print("=" * 72)
    print(f"Candidate edges: {report['candidate_edge_count']}")
    print(f"Clusters: {report['cluster_count']}")
    print(
        "Unmatched observations: "
        f"{report['unmatched_observation_count']}"
    )
    print("Outputs identical: YES")
    print()
    print(f"Reference median: {reference_median:.6f} seconds")
    print(f"Optimized median: {optimized_median:.6f} seconds")
    print(f"Measured speedup: {speedup:.2f}x")
    print()
    print(f"CSV report: {OUTPUT_CSV}")
    print(f"JSON report: {OUTPUT_JSON}")
    print()
    print("Benchmark completed successfully.")


if __name__ == "__main__":
    main()
