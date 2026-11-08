import csv
import hashlib
import zipfile
from pathlib import Path
from datetime import date

folder = Path("data/archive_downloads/nasa_firms/incoming")

for archive in sorted(folder.glob("*.zip")):
    print("\n" + "=" * 65)
    print(archive.name)
    print("=" * 65)

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    print("ZIP size:", archive.stat().st_size, "bytes")
    print("SHA256:", digest)

    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        print("ZIP integrity:", "PASS" if bad is None else f"FAIL: {bad}")

        for member in z.namelist():
            if not member.lower().endswith(".csv"):
                continue

            with z.open(member) as raw:
                import io
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig"))

                count = 0
                dates = []
                latitudes = []
                longitudes = []
                invalid = 0

                for row in reader:
                    count += 1
                    try:
                        dates.append(date.fromisoformat(row["acq_date"]))
                        latitudes.append(float(row["latitude"]))
                        longitudes.append(float(row["longitude"]))
                    except (ValueError, KeyError, TypeError):
                        invalid += 1

                print("CSV:", member)
                print("Observations:", count)
                print("Date range:", min(dates) if dates else "N/A",
                      "to", max(dates) if dates else "N/A")
                print("Unique observation dates:", len(set(dates)))
                print("Invalid rows:", invalid)

                if latitudes and longitudes:
                    print("Latitude range:", min(latitudes), "to", max(latitudes))
                    print("Longitude range:", min(longitudes), "to", max(longitudes))
