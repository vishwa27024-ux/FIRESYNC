"""Unit tests for the FIRESYNC Sensor Harmonization Engine."""

import numpy as np
import pandas as pd
import pytest

from src.science.harmonization_engine import (
    EmpiricalRatioModel,
    RandomForestHarmonizer,
    RobustHuberModel,
    TwoStageHurdleModel,
    build_daily_analytical_units,
    evaluate_harmonization_models,
    generate_harmonized_dataset,
)


@pytest.fixture
def synthetic_observations() -> pd.DataFrame:
    """Create deterministic multi-day synthetic fire observations."""
    records = []
    dates = ["2023-03-01", "2023-03-02", "2023-03-03", "2024-03-01", "2024-03-02"]

    # Day 1: Both observed (1 MODIS, 5 VIIRS)
    records.append({
        "observation_id": "M1",
        "sensor_family": "MODIS",
        "acq_date": "2023-03-01",
        "frp_mw": 25.0,
        "daynight": "D",
    })
    for i in range(5):
        records.append({
            "observation_id": f"V1_{i}",
            "sensor_family": "VIIRS",
            "acq_date": "2023-03-01",
            "frp_mw": 5.0 + i,
            "daynight": "D",
        })

    # Day 2: Only VIIRS observed (3 VIIRS)
    for i in range(3):
        records.append({
            "observation_id": f"V2_{i}",
            "sensor_family": "VIIRS",
            "acq_date": "2023-03-02",
            "frp_mw": 8.0,
            "daynight": "N",
        })

    # Day 3: Both observed (2 MODIS, 10 VIIRS)
    for i in range(2):
        records.append({
            "observation_id": f"M3_{i}",
            "sensor_family": "MODIS",
            "acq_date": "2023-03-03",
            "frp_mw": 40.0,
            "daynight": "D",
        })
    for i in range(10):
        records.append({
            "observation_id": f"V3_{i}",
            "sensor_family": "VIIRS",
            "acq_date": "2023-03-03",
            "frp_mw": 12.0,
            "daynight": "D",
        })

    # Day 4 (2024 holdout): VIIRS only
    for i in range(4):
        records.append({
            "observation_id": f"V4_{i}",
            "sensor_family": "VIIRS",
            "acq_date": "2024-03-01",
            "frp_mw": 15.0,
            "daynight": "D",
        })

    # Day 5 (2024 holdout): Both observed
    records.append({
        "observation_id": "M5",
        "sensor_family": "MODIS",
        "acq_date": "2024-03-02",
        "frp_mw": 30.0,
        "daynight": "D",
    })
    for i in range(6):
        records.append({
            "observation_id": f"V5_{i}",
            "sensor_family": "VIIRS",
            "acq_date": "2024-03-02",
            "frp_mw": 10.0,
            "daynight": "D",
        })

    return pd.DataFrame(records)


def test_build_daily_analytical_units(synthetic_observations):
    units = build_daily_analytical_units(synthetic_observations)

    assert len(units) == 5
    assert "raw_modis_count" in units.columns
    assert "raw_viirs_count" in units.columns
    assert "viirs_frp_sum" in units.columns
    assert "viirs_day_ratio" in units.columns
    assert "is_concurrent" in units.columns

    # Check Day 1 counts
    day1 = units[units["acq_date"] == "2023-03-01"].iloc[0]
    assert day1["raw_modis_count"] == 1.0
    assert day1["raw_viirs_count"] == 5.0
    assert day1["is_concurrent"] is True or day1["is_concurrent"] == 1

    # Check Day 2 (VIIRS only)
    day2 = units[units["acq_date"] == "2023-03-02"].iloc[0]
    assert day2["raw_modis_count"] == 0.0
    assert day2["raw_viirs_count"] == 3.0
    assert not day2["is_concurrent"]


def test_build_daily_analytical_units_missing_columns():
    bad_df = pd.DataFrame({"observation_id": ["A"], "sensor_family": ["MODIS"]})
    with pytest.raises(ValueError, match="Missing required columns"):
        build_daily_analytical_units(bad_df)


def test_models_non_negative_predictions(synthetic_observations):
    units = build_daily_analytical_units(synthetic_observations)
    X = units
    y = units["raw_modis_count"]

    models = [
        EmpiricalRatioModel(),
        RobustHuberModel(),
        TwoStageHurdleModel(),
        RandomForestHarmonizer(),
    ]

    for model in models:
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert np.all(preds >= 0.0), f"{model.__class__.__name__} produced negative predictions"
        assert model.residual_std >= 0.0


def test_random_forest_feature_importances(synthetic_observations):
    units = build_daily_analytical_units(synthetic_observations)
    rf = RandomForestHarmonizer()
    rf.fit(units, units["raw_modis_count"])

    assert len(rf.feature_importances_) > 0
    assert "raw_viirs_count" in rf.feature_importances_
    assert sum(rf.feature_importances_.values()) == pytest.approx(1.0, abs=1e-3)


def test_evaluate_harmonization_models(synthetic_observations):
    units = build_daily_analytical_units(synthetic_observations)
    metrics_df, models = evaluate_harmonization_models(units, split_year=2024)

    assert len(metrics_df) == 4
    required_cols = {"Model", "Train RMSE", "Val RMSE", "Train MAE", "Val MAE", "Val R²", "Val Bias", "Uncertainty (σ)"}
    assert required_cols.issubset(metrics_df.columns)
    assert len(models) == 4


def test_generate_harmonized_dataset(synthetic_observations):
    harmonized = generate_harmonized_dataset(synthetic_observations, "Two-Stage Hurdle Model")

    assert len(harmonized) == 5
    assert "calibrated_modis_equivalent" in harmonized.columns
    assert "ci_lower_95" in harmonized.columns
    assert "ci_upper_95" in harmonized.columns
    assert "data_quality_flag" in harmonized.columns

    # Prediction interval ordering check
    assert np.all(harmonized["ci_lower_95"] <= harmonized["calibrated_modis_equivalent"])
    assert np.all(harmonized["calibrated_modis_equivalent"] <= harmonized["ci_upper_95"])

    # Quality flags check
    flags = set(harmonized["data_quality_flag"])
    assert "CONCURRENT_CALIBRATED" in flags
    assert "VIIRS_IMPUTED_MODIS_EQUIVALENT" in flags


def test_harmonization_extreme_frp_edge_case():
    extreme_df = pd.DataFrame([
        {
            "observation_id": "V_EXT",
            "sensor_family": "VIIRS",
            "acq_date": "2023-01-01",
            "frp_mw": 9999.0,
            "daynight": "D",
        },
        {
            "observation_id": "M_EXT",
            "sensor_family": "MODIS",
            "acq_date": "2023-01-01",
            "frp_mw": 5000.0,
            "daynight": "D",
        },
    ])
    units = build_daily_analytical_units(extreme_df)
    harmonized = generate_harmonized_dataset(extreme_df, "Robust Log-Log Huber (Baseline)")
    assert not np.isnan(harmonized["calibrated_modis_equivalent"].iloc[0])
    assert not np.isinf(harmonized["calibrated_modis_equivalent"].iloc[0])
    assert harmonized["calibrated_modis_equivalent"].iloc[0] >= 0.0
