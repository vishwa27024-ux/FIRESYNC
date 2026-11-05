from pathlib import Path
import json
import xml.etree.ElementTree as ET

from pyproj import Geod, Transformer
from shapely.geometry import Polygon, shape
from shapely.ops import transform as project_geometry


ROOT = Path(__file__).resolve().parents[1]

KML = (
    ROOT / "data" / "boundaries" / "source"
    / "ntca_extracted" / "PA_TR_Corridor_Final.kml"
)

GEOJSON = (
    ROOT / "data" / "boundaries"
    / "bandipur_ntca_provisional.geojson"
)

GEOD = Geod(ellps="WGS84")

TRANSFORMER = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:32643",
    always_xy=True,
)


def tag_name(element):
    return element.tag.split("}")[-1]


def read_coordinates(ring):
    node = next(
        (
            element for element in ring.iter()
            if tag_name(element) == "coordinates"
            and element.text
        ),
        None,
    )

    if node is None:
        raise ValueError("LinearRing coordinates missing.")

    result = []

    for token in node.text.split():
        values = token.split(",")
        result.append(
            (float(values[0]), float(values[1]))
        )

    return result


def ring_area_km2(coordinates):
    longitudes = [point[0] for point in coordinates]
    latitudes = [point[1] for point in coordinates]

    area, _ = GEOD.polygon_area_perimeter(
        longitudes,
        latitudes,
    )

    return abs(area) / 1_000_000


def main():
    print("=" * 65)
    print("FIRESYNC | BOUNDARY AREA VERIFICATION")
    print("=" * 65)

    tree = ET.parse(KML)

    matches = []

    for placemark in tree.getroot().iter():
        if tag_name(placemark) != "Placemark":
            continue

        name = next(
            (
                child.text.strip()
                for child in placemark
                if tag_name(child) == "name" and child.text
            ),
            "",
        )

        if name.casefold() == "bandipur":
            matches.append(placemark)

    if len(matches) != 1:
        raise RuntimeError("Expected exactly one Bandipur placemark.")

    placemark = matches[0]

    reported_area = None

    for element in placemark.iter():
        if tag_name(element) in {"Data", "SimpleData"}:
            if element.attrib.get("name") == "REP_AREA":
                value = next(
                    (
                        child.text
                        for child in element.iter()
                        if tag_name(child) == "value"
                        and child.text
                    ),
                    None,
                )

                if value is None:
                    value = "".join(element.itertext()).strip()

                reported_area = float(value)

    if reported_area is None:
        raise RuntimeError("REP_AREA metadata not found.")

    polygons = []

    for node in placemark.iter():
        if tag_name(node) != "Polygon":
            continue

        outer = next(
            child for child in node
            if tag_name(child) == "outerBoundaryIs"
        )

        outer_rings = [
            element for element in outer.iter()
            if tag_name(element) == "LinearRing"
        ]

        if len(outer_rings) != 1:
            raise RuntimeError("Unexpected exterior ring structure.")

        shell = read_coordinates(outer_rings[0])

        hole_coordinates = []

        for boundary in node:
            if tag_name(boundary) != "innerBoundaryIs":
                continue

            for ring in boundary.iter():
                if tag_name(ring) == "LinearRing":
                    hole_coordinates.append(
                        read_coordinates(ring)
                    )

        exterior_polygon = Polygon(shell)

        print("\nEXTERIOR RING")
        print(f"Vertices: {len(shell)}")
        print(
            f"Geodesic area: {ring_area_km2(shell):.3f} km²"
        )

        valid_holes = []

        for index, coordinates in enumerate(
            hole_coordinates, start=1
        ):
            hole = Polygon(coordinates)

            within_exterior = exterior_polygon.contains(
                hole.representative_point()
            )

            fully_covered = exterior_polygon.covers(hole)

            print(f"\nHOLE {index}")
            print(f"Vertices: {len(coordinates)}")
            print(f"Valid ring polygon: {hole.is_valid}")
            print(f"Inside exterior: {within_exterior}")
            print(f"Fully covered by exterior: {fully_covered}")
            print(
                f"Geodesic area: "
                f"{ring_area_km2(coordinates):.6f} km²"
            )

            if not within_exterior or not fully_covered:
                raise RuntimeError(
                    f"Interior ring {index} is not fully within exterior."
                )

            valid_holes.append(hole)

        polygon = Polygon(shell, hole_coordinates)

        if not polygon.is_valid:
            raise RuntimeError("Source polygon is invalid.")

        polygons.append(polygon)

    source_geometry = polygons[0] if len(polygons) == 1 else None

    if source_geometry is None:
        raise RuntimeError(
            "Multiple polygons require separate area review."
        )

    geodesic_net = (
        ring_area_km2(shell)
        - sum(
            ring_area_km2(coordinates)
            for coordinates in hole_coordinates
        )
    )

    projected = project_geometry(
        TRANSFORMER.transform,
        source_geometry,
    )

    projected_area = projected.area / 1_000_000

    with GEOJSON.open(encoding="utf-8") as file:
        geojson = json.load(file)

    output_geometries = [
        shape(feature["geometry"])
        for feature in geojson["features"]
        if feature.get("geometry")
    ]

    if len(output_geometries) != 1:
        raise RuntimeError("Unexpected GeoJSON feature count.")

    geojson_projected = project_geometry(
        TRANSFORMER.transform,
        output_geometries[0],
    )

    geojson_area = geojson_projected.area / 1_000_000

    discrepancy = projected_area - reported_area
    discrepancy_pct = discrepancy / reported_area * 100

    print("\n" + "=" * 65)
    print("AREA COMPARISON")
    print("=" * 65)

    print(f"KML REP_AREA attribute: {reported_area:.3f} km²")
    print(f"Geodesic net area: {geodesic_net:.3f} km²")
    print(f"Projected UTM area: {projected_area:.3f} km²")
    print(f"Exported GeoJSON area: {geojson_area:.3f} km²")

    print(f"\nDifference from REP_AREA: {discrepancy:.3f} km²")
    print(f"Percentage difference: {discrepancy_pct:.3f}%")

    print("\nInterpretation:")
    print("REP_AREA units are treated as km² for comparison.")
    print("Different area methods and source metadata may differ.")
    print("This script does not certify legal boundary accuracy.")

    print("\nOriginal source files remain unchanged.")
    print("=" * 65)


if __name__ == "__main__":
    main()
