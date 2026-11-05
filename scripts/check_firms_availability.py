import pandas as pd

from src.data.firms_client import (
    SUPPORTED_PRODUCTS,
    fetch_data_availability,
)


def main():
    print("=" * 60)
    print("FIRESYNC | NASA FIRMS DATA AVAILABILITY")
    print("=" * 60)

    availability = fetch_data_availability("all")

    products = availability[
        availability["data_id"].isin(SUPPORTED_PRODUCTS)
    ].copy()

    print("\nOFFICIAL PRODUCT DATE COVERAGE\n")

    if products.empty:
        print("No requested standard-processing products were returned.")
        return

    products = products.sort_values("data_id")

    print(products.to_string(index=False))

    start = pd.Timestamp("2025-03-01")
    end = pd.Timestamp("2025-03-31")

    print("\nMARCH 2025 AVAILABILITY CHECK\n")

    for product in sorted(SUPPORTED_PRODUCTS):
        row = products[products["data_id"] == product]

        if row.empty:
            print(f"{product}: NOT REPORTED")
            continue

        minimum = row.iloc[0]["min_date"]
        maximum = row.iloc[0]["max_date"]

        overlaps = (
            pd.notna(minimum)
            and pd.notna(maximum)
            and minimum <= end
            and maximum >= start
        )

        print(
            f"{product}: "
            f"{'DATE RANGE AVAILABLE' if overlaps else 'NO OVERLAP'}"
        )

    print("\nAvailability check completed.")


if __name__ == "__main__":
    main()
