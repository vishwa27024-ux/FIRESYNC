"""Sensor Harmonization and Calibration Engine for FIRESYNC.

This module addresses PRD Requirement FR-05 by providing statistical
calibration models that harmonize MODIS and VIIRS active-fire observations
across overlapping observation periods.

The goal is not to force VIIRS counts to equal MODIS counts, but to
estimate statistically comparable fire-activity metrics while accounting
for resolution disparities, sub-pixel detection thresholds, and sensor bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURE_COLUMNS = [
    "raw_viirs_count",
    "viirs_frp_sum",
    "viirs_frp_mean",
    "viirs_frp_max",
    "viirs_day_ratio",
    "month",
]


@dataclass
class ModelMetrics:
    """Evaluation metrics for a trained calibration model."""

    model_name: str
    train_rmse: float
    val_rmse: float
    train_mae: float
    val_mae: float
    val_r2: float
    val_bias: float
    residual_std: float


def build_daily_analytical_units(observations: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw observations into daily landscape analytical units.

    Extracts multi-sensor metrics per acquisition date, including
    counts, FRP distributions, and temporal overpass indicators.
    """
    if observations.empty:
        raise ValueError("Observations DataFrame cannot be empty.")

    required = {"observation_id", "sensor_family", "acq_date", "frp_mw"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = observations.copy()
    df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.date
    df["frp_mw"] = pd.to_numeric(df["frp_mw"], errors="coerce").fillna(0.0)

    # Determine day/night indicator if present
    if "daynight" in df.columns:
        df["is_day"] = df["daynight"].astype(str).str.upper() == "D"
    else:
        df["is_day"] = True

    # Group by date and sensor family
    sensor_daily = (
        df.groupby(["acq_date", "sensor_family"])
        .agg(
            count=("observation_id", "count"),
            frp_sum=("frp_mw", "sum"),
            frp_mean=("frp_mw", "mean"),
            frp_max=("frp_mw", "max"),
            day_count=("is_day", "sum"),
        )
        .unstack(level="sensor_family", fill_value=0)
    )

    # Flatten column MultiIndex
    flat_cols = []
    for col, sensor in sensor_daily.columns:
        flat_cols.append(f"{sensor.lower()}_{col}")
    sensor_daily.columns = flat_cols
    sensor_daily = sensor_daily.reset_index()

    # Ensure required fields exist even if a sensor had 0 records
    for sensor in ["modis", "viirs"]:
        for metric in ["count", "frp_sum", "frp_mean", "frp_max", "day_count"]:
            col_name = f"{sensor}_{metric}"
            if col_name not in sensor_daily.columns:
                sensor_daily[col_name] = 0.0

    # Rename to clean analytical names
    units = pd.DataFrame(
        {
            "acq_date": pd.to_datetime(sensor_daily["acq_date"]),
            "raw_modis_count": sensor_daily["modis_count"].astype(float),
            "raw_viirs_count": sensor_daily["viirs_count"].astype(float),
            "modis_frp_sum": sensor_daily["modis_frp_sum"].astype(float),
            "viirs_frp_sum": sensor_daily["viirs_frp_sum"].astype(float),
            "viirs_frp_mean": sensor_daily["viirs_frp_mean"].astype(float),
            "viirs_frp_max": sensor_daily["viirs_frp_max"].astype(float),
            "viirs_day_count": sensor_daily["viirs_day_count"].astype(float),
        }
    )

    units["viirs_day_ratio"] = np.where(
        units["raw_viirs_count"] > 0,
        units["viirs_day_count"] / units["raw_viirs_count"],
        1.0,
    )
    units["year"] = units["acq_date"].dt.year
    units["month"] = units["acq_date"].dt.month
    try:
        units["day_of_year"] = units["acq_date"].dt.day_of_year
    except AttributeError:
        units["day_of_year"] = units["acq_date"].dt.dayofyear
    units["is_concurrent"] = (units["raw_modis_count"] > 0) & (
        units["raw_viirs_count"] > 0
    )

    return units.sort_values("acq_date").reset_index(drop=True)


class EmpiricalRatioModel:
    """Baseline 1: Constant detection ratio scaling model."""

    def __init__(self):
        self.ratio: float = 1.0
        self.residual_std: float = 1.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> EmpiricalRatioModel:
        viirs_count = X["raw_viirs_count"].values
        modis_count = y.values

        # Compute ratio on days where both observed or VIIRS observed
        mask = (viirs_count > 0) & (modis_count > 0)
        if mask.sum() > 0:
            self.ratio = float(viirs_count[mask].sum() / modis_count[mask].sum())
        else:
            self.ratio = 10.0

        preds = self.predict(X)
        self.residual_std = float(np.std(modis_count - preds, ddof=1)) if len(modis_count) > 1 else 1.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        viirs_count = X["raw_viirs_count"].values
        return np.maximum(0.0, viirs_count / max(self.ratio, 1e-6))


class RobustHuberModel:
    """Baseline 2: Log-log robust Huber regressor resisting fire-surge outliers."""

    def __init__(self):
        self.regressor = HuberRegressor(epsilon=1.35, max_iter=500)
        self.residual_std: float = 1.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RobustHuberModel:
        x_features = np.column_stack(
            [
                np.log1p(X["raw_viirs_count"].values),
                np.log1p(X["viirs_frp_sum"].values),
                np.log1p(X["viirs_frp_max"].values),
            ]
        )
        y_log = np.log1p(y.values)
        self.regressor.fit(x_features, y_log)

        preds = self.predict(X)
        residuals = y.values - preds
        self.residual_std = float(np.std(residuals, ddof=1)) if len(y) > 1 else 1.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        x_features = np.column_stack(
            [
                np.log1p(X["raw_viirs_count"].values),
                np.log1p(X["viirs_frp_sum"].values),
                np.log1p(X["viirs_frp_max"].values),
            ]
        )
        pred_log = self.regressor.predict(x_features)
        # Invert log1p and ensure non-negative
        return np.maximum(0.0, np.expm1(np.maximum(0.0, pred_log)))


class TwoStageHurdleModel:
    """Model 3: Zero-inflation hurdle model.

    Stage 1: Logistic regression estimates P(MODIS > 0 | VIIRS features).
    Stage 2: Ridge log-linear regressor estimates intensity E[MODIS | MODIS > 0].
    """

    def __init__(self):
        self.classifier = LogisticRegression(class_weight="balanced", max_iter=500)
        self.regressor = Ridge(alpha=1.0)
        self.constant_class: Optional[int] = None
        self.residual_std: float = 1.0

    def _extract_x(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [
                np.log1p(X["raw_viirs_count"].values),
                np.log1p(X["viirs_frp_sum"].values),
                np.log1p(X["viirs_frp_max"].values),
                X["viirs_day_ratio"].values,
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TwoStageHurdleModel:
        x_mat = self._extract_x(X)
        y_vals = y.values
        y_binary = (y_vals > 0).astype(int)

        # Stage 1: Classifier (or constant class if only one class exists)
        if len(np.unique(y_binary)) > 1:
            self.classifier.fit(x_mat, y_binary)
            self.constant_class = None
        else:
            self.constant_class = int(y_binary[0])

        # Stage 2: Regressor on positive days
        pos_mask = y_vals > 0
        if pos_mask.sum() >= 3:
            self.regressor.fit(x_mat[pos_mask], np.log1p(y_vals[pos_mask]))
        else:
            self.regressor.fit(x_mat, np.log1p(y_vals))

        preds = self.predict(X)
        residuals = y_vals - preds
        self.residual_std = float(np.std(residuals, ddof=1)) if len(y_vals) > 1 else 1.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        x_mat = self._extract_x(X)
        if self.constant_class is not None:
            p_detect = np.full(len(x_mat), float(self.constant_class))
        elif hasattr(self.classifier, "predict_proba"):
            p_detect = self.classifier.predict_proba(x_mat)[:, 1]
        else:
            p_detect = np.ones(len(x_mat))

        pred_intensity_log = self.regressor.predict(x_mat)
        intensity = np.maximum(0.0, np.expm1(np.maximum(0.0, pred_intensity_log)))
        return np.maximum(0.0, p_detect * intensity)


class RandomForestHarmonizer:
    """Model 4: Multi-feature Random Forest Regressor."""

    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=2,
            random_state=42,
        )
        self.residual_std: float = 1.0
        self.feature_importances_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RandomForestHarmonizer:
        features = X[FEATURE_COLUMNS].fillna(0.0)
        y_log = np.log1p(y.values)
        self.model.fit(features, y_log)

        for col, imp in zip(FEATURE_COLUMNS, self.model.feature_importances_):
            self.feature_importances_[col] = float(imp)

        preds = self.predict(X)
        residuals = y.values - preds
        self.residual_std = float(np.std(residuals, ddof=1)) if len(y) > 1 else 1.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        features = X[FEATURE_COLUMNS].fillna(0.0)
        pred_log = self.model.predict(features)
        return np.maximum(0.0, np.expm1(np.maximum(0.0, pred_log)))


def evaluate_harmonization_models(
    units: pd.DataFrame,
    split_year: int = 2024,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Evaluate all 4 harmonization models on independent chronological split.

    Training period: year < split_year.
    Validation period: year >= split_year.
    """
    if len(units) < 3:
        train_units = units
        val_units = units
    else:
        train_mask = units["year"] < split_year
        val_mask = units["year"] >= split_year

        if train_mask.sum() == 0 or val_mask.sum() == 0:
            # Fallback to 80/20 chronological split if split_year isolates all data
            split_idx = max(1, int(len(units) * 0.8))
            train_units = units.iloc[:split_idx]
            val_units = units.iloc[split_idx:]
        else:
            train_units = units[train_mask]
            val_units = units[val_mask]

    x_train = train_units
    y_train = train_units["raw_modis_count"]

    x_val = val_units
    y_val = val_units["raw_modis_count"]

    models = {
        "Empirical Area Ratio (Baseline)": EmpiricalRatioModel(),
        "Robust Log-Log Huber (Baseline)": RobustHuberModel(),
        "Two-Stage Hurdle Model": TwoStageHurdleModel(),
        "Random Forest Harmonizer": RandomForestHarmonizer(),
    }

    metrics_list = []
    fitted_models: Dict[str, Any] = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        fitted_models[name] = model

        train_preds = model.predict(x_train)
        val_preds = model.predict(x_val)

        train_rmse = float(np.sqrt(mean_squared_error(y_train, train_preds)))
        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))

        train_mae = float(mean_absolute_error(y_train, train_preds))
        val_mae = float(mean_absolute_error(y_val, val_preds))

        val_r2 = float(r2_score(y_val, val_preds)) if len(y_val) > 1 and np.var(y_val) > 0 else 0.0
        val_bias = float(np.mean(val_preds - y_val.values))

        metrics_list.append(
            {
                "Model": name,
                "Train RMSE": round(train_rmse, 3),
                "Val RMSE": round(val_rmse, 3),
                "Train MAE": round(train_mae, 3),
                "Val MAE": round(val_mae, 3),
                "Val R²": round(val_r2, 3),
                "Val Bias": round(val_bias, 3),
                "Uncertainty (σ)": round(model.residual_std, 3),
            }
        )

    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df, fitted_models


def generate_harmonized_dataset(
    observations: pd.DataFrame,
    preferred_model_name: str = "Two-Stage Hurdle Model",
) -> pd.DataFrame:
    """Generate a calibrated historical dataset with 95% uncertainty intervals.

    Produces daily analytical records with raw observations alongside
    calibrated MODIS-equivalent counts and data quality attribution.
    """
    units = build_daily_analytical_units(observations)
    metrics_df, models = evaluate_harmonization_models(units)

    selected_model = models.get(preferred_model_name)
    if selected_model is None:
        selected_model = next(iter(models.values()))
        preferred_model_name = next(iter(models.keys()))

    # Full fit on entire dataset to provide historical calibration
    selected_model.fit(units, units["raw_modis_count"])
    calibrated_counts = selected_model.predict(units)

    sigma = selected_model.residual_std
    ci_lower = np.maximum(0.0, calibrated_counts - 1.96 * sigma)
    ci_upper = calibrated_counts + 1.96 * sigma

    # Assign data quality flags based on observation presence
    quality_flags = []
    for _, row in units.iterrows():
        m = row["raw_modis_count"]
        v = row["raw_viirs_count"]
        if m > 0 and v > 0:
            quality_flags.append("CONCURRENT_CALIBRATED")
        elif v > 0 and m == 0:
            quality_flags.append("VIIRS_IMPUTED_MODIS_EQUIVALENT")
        elif m > 0 and v == 0:
            quality_flags.append("MODIS_ONLY_OBSERVED")
        else:
            quality_flags.append("NO_DETECTION")

    result = units.copy()
    result["calibrated_modis_equivalent"] = np.round(calibrated_counts, 2)
    result["ci_lower_95"] = np.round(ci_lower, 2)
    result["ci_upper_95"] = np.round(ci_upper, 2)
    result["calibration_model"] = preferred_model_name
    result["data_quality_flag"] = quality_flags

    return result
