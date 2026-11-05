from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

WEST, SOUTH, EAST, NORTH = 76.15, 11.55, 76.91, 11.96
START_DATE = pd.Timestamp("2025-03-01")
END_DATE = pd.Timestamp("2025-03-05")

REQUIRED_COLUMNS = {
    "latitude",
    "longitude",
    "acq_date",
}


def main():
    files = sorted(RAW_DIR.glob("*/*/*.csv"))

    print("=" * 65)
    print("FIRESYNC | RAW NASA FIRMS DATA AUDIT")
    print("=" * 65)

    if not files:
        raise SystemExit("No raw CSV files found.")

    total_records = 0

    for path in files:
        print(f"\nPRODUCT: {path.parent.name}")
        print(f"FILE: {path.name}")

        data = pd.read_csv(path)
        total_records += len(data)

        print(f"RECORDS: {len(data)}")
        print(f"COLUMNS: {', '.join(data.columns)}")

        missing = REQUIRED_COLUMNS - set(data.columns)

        if missing:
            print(f"MISSING REQUIRED FIELDS: {sorted(missing)}")
            continue

        lat = pd.to_numeric(data["latitude"], errors="coerce")
        lon = pd.to_numeric(data["longitude"], errors="coerce")
        dates = pd.to_datetime(data["acq_date"], errors="coerce")

        valid_coords = (
            lat.between(-90, 90)
            & lon.between(-180, 180)
        )

        inside_bbox = (
            lat.between(SOUTH, NORTH)
            & lon.between(WEST, EAST)
        )

        date_valid = dates.between(START_DATE, END_DATE)

        print(f"VALID COORDINATES: {valid_coords.sum()}")
        print(f"INVALID COORDINATES: {(~valid_coords).sum()}")
        print(f"INSIDE QUERY BBOX: {inside_bbox.sum()}")
        print(f"OUTSIDE QUERY BBOX: {(~inside_bbox).sum()}")
        print(f"DATES WITHIN REQUEST: {date_valid.sum()}")
        print(f"INVALID DATES: {dates.isna().sum()}")

        if dates.notna().any():
            print(
                "DATE RANGE:",
                dates.min().date(),
                "to",
                dates.max().date(),
            )

        for column in [
            "satellite",
            "instrument",
            "confidence",
            "version",
            "daynight",
        ]:
            if column in data.columns:
                values = data[column].dropna().astype(str).unique()
                print(f"{column.upper()} VALUES: {list(values)}")

        preview_columns = [
            col for col in [
                "latitude",
                "longitude",
                "acq_date",
                "acq_time",
                "satellite",
                "instrument",
                "confidence",
                "frp",
                "daynight",
            ]
            if col in data.columns
        ]

        print("\nOBSERVATION PREVIEW:")
        print(data[preview_columns].head(10).to_string(index=False))

    print("\n" + "=" * 65)
    print(f"TOTAL RAW RECORDS: {total_records}")
    print(f"FILES INSPECTED: {len(files)}")
    print("RAW FILES WERE NOT MODIFIED.")
    print("=" * 65)


if __name__ == "__main__":
    main()
