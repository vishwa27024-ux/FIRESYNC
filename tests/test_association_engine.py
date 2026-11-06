
import pandas as pd
import pytest

from src.science.association_engine import build_associations


def make_observations():
    """Create a small, deterministic synthetic dataset."""

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


def test_many_to_many_cluster():
    result = build_associations(make_observations())

    assert len(result.edges) == 4
    assert len(result.clusters) == 1

    cluster = result.clusters.iloc[0]

    assert cluster["modis_count"] == 2
    assert cluster["viirs_count"] == 2
    assert cluster["candidate_edge_count"] == 4
    assert cluster["ambiguity_type"] == "many_to_many"


def test_unmatched_observation_is_preserved():
    result = build_associations(make_observations())

    assert "M3" in result.unmatched["observation_id"].tolist()


def test_tighter_distance_threshold_reduces_edges():
    observations = make_observations()

    broad = build_associations(
        observations,
        distance_threshold_km=1.0,
        time_window_minutes=60,
    )

    tight = build_associations(
        observations,
        distance_threshold_km=0.1,
        time_window_minutes=60,
    )

    assert len(tight.edges) < len(broad.edges)


def test_time_threshold_excludes_old_observations():
    observations = make_observations()

    result = build_associations(
        observations,
        distance_threshold_km=1.0,
        time_window_minutes=5,
    )

    assert len(result.edges) == 0
    assert len(result.clusters) == 0


def test_no_cross_sensor_candidate_when_only_one_sensor_exists():
    observations = make_observations()
    observations = observations[
        observations["sensor_family"] == "MODIS"
    ]

    result = build_associations(observations)

    assert len(result.edges) == 0
    assert len(result.clusters) == 0
    assert len(result.unmatched) == len(observations)


def test_duplicate_observation_ids_are_rejected():
    observations = make_observations()
    duplicate = observations.iloc[[0]].copy()
    observations = pd.concat(
        [observations, duplicate],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="must be unique"):
        build_associations(observations)


def test_invalid_coordinates_are_rejected():
    observations = make_observations()
    observations.loc[0, "latitude"] = 95

    with pytest.raises(ValueError, match="Latitude"):
        build_associations(observations)
