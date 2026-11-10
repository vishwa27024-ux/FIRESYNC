
"""Audit temporal and sensor coverage of the FIRESYNC historical dataset."""

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "firms_archive_normalized_2021_2025.parquet"
)

OUTPUT_DIR = ROOT / "data" / "interim" / "dataset_coverage_audit"


def save_table(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(OUTPUT_DIR / filename, index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Historical dataset not found: {INPUT_FILE}"
        )

    df = pd.read_parquet(INPUT_FILE)

    required = [
        "observation_id",
        "sensor_family",
        "satellite_platform",
        "acq_datetime_utc",
    ]

    missing_columns = [
        column for column in required if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["acq_datetime_utc"] = pd.to_datetime(
        df["acq_datetime_utc"],
        utc=True,
        errors="coerce",
    )

    invalid_timestamps = int(
        df["acq_datetime_utc"].isna().sum()
    )

    valid = df.dropna(
        subset=["acq_datetime_utc"]
    ).copy()

    if valid.empty:
        raise ValueError("No valid acquisition timestamps found.")

    valid["year"] = valid["acq_datetime_utc"].dt.year
    valid["month"] = valid["acq_datetime_utc"].dt.month
    valid["year_month"] = (
        valid["acq_datetime_utc"].dt.to_period("M").astype(str)
    )

    # --------------------------------------------------------
    # 1. Annual coverage
    # --------------------------------------------------------

    annual = (
        valid.groupby("year")
        .agg(
            observations=("observation_id", "size"),
            unique_days=("acq_datetime_utc", lambda x: x.dt.date.nunique()),
            sensors=("sensor_family", "nunique"),
            platforms=("satellite_platform", "nunique"),
        )
        .reset_index()
    )

    annual["percentage"] = (
        annual["observations"] / len(valid) * 100
    ).round(2)

    save_table(annual, "annual_coverage.csv")

    # --------------------------------------------------------
    # 2. Monthly coverage across all years
    # --------------------------------------------------------

    monthly = (
        valid.groupby("month")
        .agg(
            observations=("observation_id", "size"),
            years_present=("year", "nunique"),
        )
        .reindex(range(1, 13), fill_value=0)
        .rename_axis("month")
        .reset_index()
    )

    monthly["percentage"] = (
        monthly["observations"] / len(valid) * 100
    ).round(2)

    month_names = {
        month: pd.Timestamp(2000, month, 1).strftime("%B")
        for month in range(1, 13)
    }

    monthly["month_name"] = monthly["month"].map(month_names)

    save_table(monthly, "monthly_coverage.csv")

    # --------------------------------------------------------
    # 3. Year-month coverage, including zero-record months
    # --------------------------------------------------------

    first_year = int(valid["year"].min())
    last_year = int(valid["year"].max())

    complete_months = pd.MultiIndex.from_product(
        [
            range(first_year, last_year + 1),
            range(1, 13),
        ],
        names=["year", "month"],
    )

    year_month = (
        valid.groupby(["year", "month"])
        .size()
        .reindex(complete_months, fill_value=0)
        .rename("observations")
        .reset_index()
    )

    year_month["year_month"] = (
        year_month["year"].astype(str)
        + "-"
        + year_month["month"].astype(str).str.zfill(2)
    )

    year_month["retrieved_records_present"] = (
        year_month["observations"] > 0
    )

    save_table(year_month, "year_month_coverage.csv")

    # --------------------------------------------------------
    # 4. Sensor coverage
    # --------------------------------------------------------

    sensor = (
        valid.groupby("sensor_family")
        .agg(
            observations=("observation_id", "size"),
            platforms=("satellite_platform", "nunique"),
            first_observation=("acq_datetime_utc", "min"),
            last_observation=("acq_datetime_utc", "max"),
        )
        .reset_index()
    )

    sensor["percentage"] = (
        sensor["observations"] / len(valid) * 100
    ).round(2)

    save_table(sensor, "sensor_coverage.csv")

    # --------------------------------------------------------
    # 5. Satellite platform coverage
    # --------------------------------------------------------

    platform = (
        valid.groupby(
            ["sensor_family", "satellite_platform"],
            dropna=False,
        )
        .agg(
            observations=("observation_id", "size"),
            first_observation=("acq_datetime_utc", "min"),
            last_observation=("acq_datetime_utc", "max"),
            unique_days=("acq_datetime_utc", lambda x: x.dt.date.nunique()),
        )
        .reset_index()
    )

    platform["percentage"] = (
        platform["observations"] / len(valid) * 100
    ).round(2)

    save_table(platform, "platform_coverage.csv")

    # --------------------------------------------------------
    # 6. Sensor-by-month distribution
    # --------------------------------------------------------

    sensor_month = (
        valid.groupby(["sensor_family", "month"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(1, 13), fill_value=0)
        .reset_index()
    )

    save_table(sensor_month, "sensor_month_coverage.csv")

    # --------------------------------------------------------
    # 7. Platform-by-year distribution
    # --------------------------------------------------------

    platform_year = (
        valid.groupby(["satellite_platform", "year"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    save_table(platform_year, "platform_year_coverage.csv")

    # --------------------------------------------------------
    # 8. Data quality and coverage summary
    # --------------------------------------------------------

    jan_to_mar = int(
        valid["month"].isin([1, 2, 3]).sum()
    )

    summary = {
        "dataset": INPUT_FILE.name,
        "total_input_observations": int(len(df)),
        "valid_timestamp_observations": int(len(valid)),
        "invalid_timestamp_observations": invalid_timestamps,
        "unique_observation_ids": int(
            valid["observation_id"].nunique()
        ),
        "duplicate_observation_ids": int(
            valid["observation_id"].duplicated().sum()
        ),
        "first_acquisition_utc": str(
            valid["acq_datetime_utc"].min()
        ),
        "last_acquisition_utc": str(
            valid["acq_datetime_utc"].max()
        ),
        "year_range": [first_year, last_year],
        "sensor_count": int(
            valid["sensor_family"].nunique()
        ),
        "platform_count": int(
            valid["satellite_platform"].nunique()
        ),
        "january_to_march_observations": jan_to_mar,
        "january_to_march_percentage": round(
            jan_to_mar / len(valid) * 100, 2
        ),
        "year_month_periods": int(len(year_month)),
        "year_month_periods_with_records": int(
            (year_month["observations"] > 0).sum()
        ),
        "year_month_periods_with_zero_retrieved_records": int(
            (year_month["observations"] == 0).sum()
        ),
        "interpretation_warning": (
            "A month with zero retrieved records does not prove "
            "there was no fire or no satellite observation coverage."
        ),
    }

    with open(
        OUTPUT_DIR / "coverage_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    # --------------------------------------------------------
    # 9. Terminal report
    # --------------------------------------------------------

    print("\n" + "=" * 68)
    print("FIRESYNC | HISTORICAL DATASET COVERAGE AUDIT")
    print("=" * 68)

    print(f"\nTotal observations: {len(df):,}")
    print(f"Valid timestamps: {len(valid):,}")
    print(f"Invalid timestamps: {invalid_timestamps}")
    print(
        "Unique observation IDs:",
        valid["observation_id"].nunique(),
    )

    print("\nANNUAL COVERAGE")
    print(annual.to_string(index=False))

    print("\nMONTHLY COVERAGE")
    print(
        monthly[
            ["month_name", "observations", "percentage", "years_present"]
        ].to_string(index=False)
    )

    print("\nSENSOR COVERAGE")
    print(sensor.to_string(index=False))

    print("\nPLATFORM COVERAGE")
    print(platform.to_string(index=False))

    print("\nSENSOR-BY-MONTH COVERAGE")
    print(sensor_month.to_string(index=False))

    print("\nYEAR-MONTH PERIODS WITH ZERO RETRIEVED RECORDS")
    print(
        year_month.loc[
            year_month["observations"] == 0,
            ["year_month", "observations"],
        ].to_string(index=False)
    )

    print("\n" + "-" * 68)
    print("IMPORTANT INTERPRETATION")
    print(
        "Zero retrieved records do not prove no fire "
        "or absence of satellite coverage."
    )
    print(f"\nReports saved to: {OUTPUT_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    main()
