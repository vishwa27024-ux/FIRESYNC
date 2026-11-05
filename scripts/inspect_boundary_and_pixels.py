from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

KML = (
    ROOT / "data" / "boundaries" / "source"
    / "ntca_extracted" / "PA_TR_Corridor_Final.kml"
)

QC = ROOT / "data" / "interim" / "spatial_quality_control.csv"

PRODUCT_TO_FOLDER = {
    "MODIS_SP": "modis",
    "VIIRS_SNPP_SP": "viirs",
    "VIIRS_NOAA20_SP": "viirs",
    "VIIRS_NOAA21_SP": "viirs",
}


def tag_name(element):
    return element.tag.split("}")[-1]


def direct_name(element):
    for child in element:
        if tag_name(child) == "name" and child.text:
            return child.text.strip()
    return None


def inspect_kml_lineage():
    tree = ET.parse(KML)
    root = tree.getroot()
    found = []

    def walk(element, ancestry):
        current = ancestry
        tag = tag_name(element)

        if tag in {"Folder", "Document"}:
            name = direct_name(element)
            if name:
                current = ancestry + [name]

        if tag == "Placemark" and (
            direct_name(element) or ""
        ).casefold() == "bandipur":

            description = ""

            for child in element:
                if tag_name(child) == "description":
                    description = "".join(child.itertext())
                    break

            extended = {}

            for child in element.iter():
                if tag_name(child) in {"Data", "SimpleData"}:
                    key = child.attrib.get("name", "unnamed")
                    value = "".join(child.itertext()).strip()
                    extended[key] = value

            polygons = [
                child for child in element.iter()
                if tag_name(child) == "Polygon"
            ]

            found.append({
                "ancestry": current,
                "description": description[:700],
                "extended": extended,
                "polygon_count": len(polygons),
            })

        for child in element:
            walk(child, current)

    walk(root, [])

    print("=" * 65)
    print("FIRESYNC | BOUNDARY SOURCE LINEAGE")
    print("=" * 65)

    print(f"Bandipur placemarks found: {len(found)}")

    for item in found:
        print("\nFolder hierarchy:")
        print(" > ".join(item["ancestry"]))

        print("\nDescription:")
        print(item["description"] or "No description")

        print("\nExtended data:")
        print(item["extended"] or "No extended attributes")

        print("\nPolygon count:", item["polygon_count"])


def inspect_pixel_attributes():
    if not QC.exists():
        raise FileNotFoundError(QC)

    audit = pd.read_csv(QC)

    cache = {}
    records = []

    for _, item in audit.iterrows():
        product = item["product"]

        if product not in PRODUCT_TO_FOLDER:
            continue

        source_file = item["source_file"]

        if product not in cache:
            path = (
                ROOT / "data" / "raw"
                / PRODUCT_TO_FOLDER[product]
                / product / source_file
            )
            cache[product] = pd.read_csv(path)

        raw = cache[product]

        source_row = int(item["source_row"])
        observation = raw.iloc[source_row]

        records.append({
            "product": product,
            "latitude": item["latitude"],
            "longitude": item["longitude"],
            "distance_to_boundary_km": item[
                "distance_to_boundary_km"
            ],
            "classification": item["classification"],
            "scan_raw": observation.get("scan"),
            "track_raw": observation.get("track"),
            "confidence": observation.get("confidence"),
            "frp": observation.get("frp"),
            "version": observation.get("version"),
            "acq_date": observation.get("acq_date"),
            "acq_time": observation.get("acq_time"),
        })

    report = pd.DataFrame(records)
    report = report.sort_values("distance_to_boundary_km")

    print("\n" + "=" * 65)
    print("FIRESYNC | NEAR-BOUNDARY OBSERVATION ATTRIBUTES")
    print("=" * 65)

    columns = [
        "product",
        "latitude",
        "longitude",
        "distance_to_boundary_km",
        "scan_raw",
        "track_raw",
        "confidence",
        "frp",
        "acq_date",
    ]

    print(report[columns].to_string(index=False))

    print("\nNo pixel footprints have been inferred.")
    print("Scan and track are displayed in their original units.")


def main():
    inspect_kml_lineage()
    inspect_pixel_attributes()


if __name__ == "__main__":
    main()
