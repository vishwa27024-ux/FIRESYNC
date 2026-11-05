from src.data.firms_client import (
    SUPPORTED_PRODUCTS,
    download_area_csv,
)

AREA = {
    "west": 76.15,
    "south": 11.55,
    "east": 76.91,
    "north": 11.96,
}

START_DATE = "2025-03-01"
DAY_RANGE = 5


def main():
    print("=" * 60)
    print("FIRESYNC | FIRST NASA FIRMS DATA ACQUISITION")
    print("=" * 60)

    failures = []

    for source in sorted(SUPPORTED_PRODUCTS):
        print(f"\nDownloading: {source}")

        try:
            path, data = download_area_csv(
                source=source,
                start_date=START_DATE,
                day_range=DAY_RANGE,
                **AREA,
            )

            print(f"Records received: {len(data):,}")
            print(f"Raw file saved: {path}")

            if data.empty:
                print("No detections returned for this period.")

        except Exception as exc:
            failures.append((source, str(exc)))
            print(f"Download failed: {exc}")

    print("\nACQUISITION SUMMARY")

    for source, error in failures:
        print(f"{source}: FAILED - {error}")

    if not failures:
        print("All four product requests completed successfully.")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
