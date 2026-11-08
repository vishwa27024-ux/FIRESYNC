
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from src.data.firms_client import (
    get_map_key,
    download_area_csv,
)


ROOT = Path(__file__).resolve().parents[1]

SOURCE = "MODIS_SP"
START_DATE = "2021-01-01"
DAY_RANGE = 5

AREA = {
    "west": 76.15,
    "south": 11.55,
    "east": 76.91,
    "north": 11.96,
}


def get_transaction_status():
    key = quote(get_map_key(), safe="")

    url = (
        "https://firms.modaps.eosdis.nasa.gov/"
        "mapserver/mapkey_status/"
        f"?MAP_KEY={key}"
    )

    request = Request(
        url,
        headers={"User-Agent": "FIRESYNC/0.1"},
    )

    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    output_dir = (
        ROOT / "data" / "raw" / "modis" / SOURCE
    )

    filename = (
        f"{SOURCE}_{START_DATE}_to_2021-01-05.csv"
    )

    csv_path = output_dir / filename
    metadata_path = csv_path.with_suffix(".json")

    # Never repeat this test request automatically.
    if csv_path.exists() or metadata_path.exists():
        raise FileExistsError(
            "Preflight output already exists. "
            "Refusing to issue another request."
        )

    before = get_transaction_status()

    limit = int(before["transaction_limit"])
    current = int(before["current_transactions"])

    print("=" * 65)
    print("FIRESYNC | NASA FIRMS TRANSACTION PREFLIGHT")
    print("=" * 65)
    print(f"Product: {SOURCE}")
    print(f"Period: {START_DATE} to 2021-01-05")
    print(f"Transactions before request: {current}")
    print(f"Transaction limit: {limit}")

    if limit - current < 500:
        raise RuntimeError(
            "Insufficient transaction headroom for a safe "
            "preflight. Wait for the NASA interval to reset."
        )

    print("\nIssuing exactly one five-day historical request...")

    path, observations = download_area_csv(
        source=SOURCE,
        start_date=START_DATE,
        day_range=DAY_RANGE,
        **AREA,
    )

    after = get_transaction_status()

    current_after = int(after["current_transactions"])

    print("\nPREFLIGHT RESULTS")
    print("-" * 65)
    print(f"Downloaded file: {path.name}")
    print(f"Returned observations: {len(observations)}")
    print(f"Transactions before: {current}")
    print(f"Transactions after: {current_after}")

    if current_after >= current:
        print(
            f"Measured transaction difference: "
            f"{current_after - current}"
        )
    else:
        print(
            "Transaction interval appears to have reset. "
            "Exact request cost could not be calculated."
        )

    print("\nThe five-day sample has been preserved.")
    print("No other historical requests were issued.")


if __name__ == "__main__":
    main()
