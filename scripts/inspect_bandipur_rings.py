from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]

KML = (
    ROOT / "data" / "boundaries" / "source"
    / "ntca_extracted" / "PA_TR_Corridor_Final.kml"
)


def local_name(tag):
    return tag.split("}")[-1]


def main():
    tree = ET.parse(KML)
    root = tree.getroot()

    placemarks = [
        e for e in root.iter()
        if local_name(e.tag) == "Placemark"
    ]

    matches = []

    for placemark in placemarks:
        name = next(
            (
                e.text.strip()
                for e in placemark
                if local_name(e.tag) == "name" and e.text
            ),
            "",
        )

        if name.casefold() == "bandipur":
            matches.append(placemark)

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one Bandipur placemark, found {len(matches)}"
        )

    polygons = [
        e for e in matches[0].iter()
        if local_name(e.tag) == "Polygon"
    ]

    print("=" * 60)
    print("FIRESYNC | KML RING STRUCTURE DIAGNOSTIC")
    print("=" * 60)

    for polygon_index, polygon in enumerate(polygons, start=1):

        print(f"\nPOLYGON {polygon_index}")

        for boundary_type in (
            "outerBoundaryIs",
            "innerBoundaryIs",
        ):
            boundaries = [
                e for e in polygon
                if local_name(e.tag) == boundary_type
            ]

            for boundary_index, boundary in enumerate(
                boundaries, start=1
            ):
                coordinates = [
                    e for e in boundary.iter()
                    if local_name(e.tag) == "coordinates"
                    and e.text
                ]

                linear_rings = [
                    e for e in boundary.iter()
                    if local_name(e.tag) == "LinearRing"
                ]

                print(
                    f"\n{boundary_type} #{boundary_index}"
                )
                print(
                    f"LinearRing elements: {len(linear_rings)}"
                )
                print(
                    f"Coordinates elements: {len(coordinates)}"
                )

                for index, element in enumerate(
                    coordinates, start=1
                ):
                    tokens = element.text.split()

                    points = []
                    for token in tokens:
                        values = token.split(",")
                        points.append(
                            (float(values[0]), float(values[1]))
                        )

                    print(f"\n  Coordinate set {index}")
                    print(f"  Number of vertices: {len(points)}")
                    print(f"  First point: {points[0]}")
                    print(f"  Last point: {points[-1]}")
                    print(
                        "  Closed ring:",
                        points[0] == points[-1],
                    )
                    print(
                        "  Bounds:",
                        (
                            min(p[0] for p in points),
                            min(p[1] for p in points),
                            max(p[0] for p in points),
                            max(p[1] for p in points),
                        ),
                    )

    print("\nDIAGNOSTIC COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
