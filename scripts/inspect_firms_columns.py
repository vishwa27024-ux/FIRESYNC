import csv
import io
import zipfile
from pathlib import Path

folder = Path("data/archive_downloads/nasa_firms/incoming")

for archive in sorted(folder.glob("*.zip")):
    print("\n" + "=" * 70)
    print(archive.name)
    print("=" * 70)

    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            if not member.lower().endswith(".csv"):
                continue

            with z.open(member) as raw:
                reader = csv.DictReader(
                    io.TextIOWrapper(raw, encoding="utf-8-sig")
                )
                print("\nCOLUMNS:")
                print(reader.fieldnames)

                print("\nFIRST TWO OBSERVATIONS:")
                for i, row in enumerate(reader):
                    if i == 2:
                        break
                    print(row)
