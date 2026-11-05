from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]

KML_FILE = (
    ROOT
    / "data"
    / "boundaries"
    / "source"
    / "ntca_extracted"
    / "PA_TR_Corridor_Final.kml"
)


def local_name(tag):
    return tag.split("}")[-1]


def inspect_kml():
    if not KML_FILE.exists():
        raise FileNotFoundError(f"KML not found: {KML_FILE}")

    print("=" * 65)
    print("FIRESYNC | NTCA BOUNDARY INSPECTION")
    print("=" * 65)

    print(f"\nFile: {KML_FILE}")
    print(f"Size: {KML_FILE.stat().st_size:,} bytes")

    tree = ET.parse(KML_FILE)
    root = tree.getroot()

    placemarks = [
        element
        for element in root.iter()
        if local_name(element.tag) == "Placemark"
    ]

    print(f"\nTotal placemarks: {len(placemarks)}")

    matches = []

    for index, placemark in enumerate(placemarks, start=1):

        name = next(
            (
                child.text.strip()
                for child in placemark
                if local_name(child.tag) == "name"
                and child.text
            ),
            "UNNAMED",
        )

        geometry_types = sorted({
            local_name(element.tag)
            for element in placemark.iter()
            if local_name(element.tag) in {
                "Polygon",
                "MultiGeometry",
                "LineString",
                "Point",
            }
        })

        if "bandipur" in name.lower():
            matches.append(
                (index, name, geometry_types)
            )

    print("\nBANDIPUR MATCHES")
    print("-" * 65)

    if matches:
        for index, name, geometry in matches:
            print(f"Placemark: {index}")
            print(f"Name: {name}")
            print(f"Geometry: {geometry}")
            print()
    else:
        print("No exact Bandipur name match found.")

        print("\nAvailable placemark names (first 60):")

        for placemark in placemarks[:60]:
            name = next(
                (
                    child.text.strip()
                    for child in placemark
                    if local_name(child.tag) == "name"
                    and child.text
                ),
                "UNNAMED",
            )
            print("-", name)

    print("=" * 65)


if __name__ == "__main__":
    inspect_kml()
