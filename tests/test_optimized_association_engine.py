
import pandas as pd
import pytest

from src.science.association_engine import build_associations
from src.science.optimized_association_engine import (
    build_associations_optimized,
)


def make_observations():
    return pd.DataFrame([
        {
            "observation_id": "M1",
            "sensor_family": "MODIS",
            "satellite_platform": "Aqua",
            "latitude": 11.5640,
            "longitude": 76.6450,
            "acq_datetime_utc": "2025-03-30T09:08:00Z",
        },
        {
            "observation_id": "M2",
            "sensor_family": "MODIS",
            "satellite_platform": "Aqua",
            "latitude": 11.5645,
            "longitude": 76.6470,
            "acq_datetime_utc": "2025-03-30T09:08:00Z",
        },
        {
            "observation_id": "V1",
            "sensor_family": "VIIRS",
            "satellite_platform": "NOAA-20",
            "latitude": 11.5642,
            "longitude": 76.6455,
            "acq_datetime_utc": "2025-03-30T08:51:00Z",
        },
        {
            "observation_id": "V2",
            "sensor_family": "VIIRS",
            "satellite_platform": "NOAA-20",
            "latitude": 11.5647,
            "longitude": 76.6475,
            "acq_datetime_utc": "2025-03-30T08:51:00Z",
        },
        {
            "observation_id": "M3",
            "sensor_family": "MODIS",
            "satellite_platform": "Terra",
            "latitude": 11.9000,
            "longitude": 76.9000,
            "acq_datetime_utc": "2025-03-30T09:08:00Z",
        },
    ])


@pytest.mark.parametrize("distance_km", [0.5, 1.0, 2.0, 5.0])
@pytest.mark.parametrize("time_minutes", [30, 60, 90, 120])
def test_optimized_matches_reference(
    distance_km,
    time_minutes,
):
    observations = make_observations()

    reference = build_associations(
        observations,
        distance_threshold_km=distance_km,
        time_window_minutes=time_minutes,
    )

    optimized = build_associations_optimized(
        observations,
        distance_threshold_km=distance_km,
        time_window_minutes=time_minutes,
    )

    reference_pairs = set(zip(
        reference.edges["modis_observation_id"],
        reference.edges["viirs_observation_id"],
    ))

    optimized_pairs = set(zip(
        optimized.edges["modis_observation_id"],
        optimized.edges["viirs_observation_id"],
    ))

    assert optimized_pairs == reference_pairs

    reference_distances = {
        (row.modis_observation_id, row.viirs_observation_id):
        row.distance_km
        for row in reference.edges.itertuples()
    }

    optimized_distances = {
        (row.modis_observation_id, row.viirs_observation_id):
        row.distance_km
        for row in optimized.edges.itertuples()
    }

    assert optimized_distances.keys() == reference_distances.keys()

    for pair in reference_distances:
        assert optimized_distances[pair] == pytest.approx(
            reference_distances[pair],
            abs=1e-9,
        )

    assert len(optimized.clusters) == len(reference.clusters)
    assert len(optimized.unmatched) == len(reference.unmatched)

    assert set(optimized.unmatched["observation_id"]) == set(
        reference.unmatched["observation_id"]
    )

    assert (
        optimized.clusters["ambiguity_type"].tolist()
        == reference.clusters["ambiguity_type"].tolist()
    )


def test_optimized_engine_preserves_many_to_many_cluster():
    result = build_associations_optimized(make_observations())

    assert len(result.edges) == 4
    assert len(result.clusters) == 1

    cluster = result.clusters.iloc[0]

    assert cluster["ambiguity_type"] == "many_to_many"
    assert cluster["modis_count"] == 2
    assert cluster["viirs_count"] == 2


def test_optimized_engine_handles_no_viirs():
    observations = make_observations()
    observations = observations[
        observations["sensor_family"] == "MODIS"
    ]

    result = build_associations_optimized(observations)

    assert result.edges.empty
    assert result.clusters.empty
    assert len(result.unmatched) == len(observations)
