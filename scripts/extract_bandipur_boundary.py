from pathlib import Path
from datetime import datetime, timezone
import json
import math
import xml.etree.ElementTree as ET

from pyproj import Transformer
from shapely.geometry import Polygon, box, mapping
from shapely.ops import transform, unary_union
from shapely.validation import explain_validity


ROOT = Path(__file__).resolve().parents[1]

KML_PATH = (
    ROOT / "data" / "boundaries" / "source"
    / "ntca_extracted" / "PA_TR_Corridor_Final.kml"
)

OUTPUT_PATH = (
    ROOT / "data" / "boundaries"
    / "bandipur_ntca_provisional.geojson"
)

METADATA_PATH = OUTPUT_PATH.with_suffix(".metadata.json")

QUERY_BBOX = (76.15, 11.55, 76.91, 11.96)


def local_name(tag):
    return tag.split("}")[-1]


def read_linear_ring(linear_ring, label):
    """Read one KML LinearRing as longitude/latitude coordinates."""

    coordinate_nodes = [
        element for element in linear_ring.iter()
        if local_name(element.tag) == "coordinates"
        and element.text
    ]

    if len(coordinate_nodes) != 1:
        raise ValueError(
            f"{label}: expected one coordinates element, "
            f"found {len(coordinate_nodes)}"
        )

    coordinates = []

    for token in coordinate_nodes[0].text.split():
        values = token.split(",")

        if len(values) not in (2, 3):
            raise ValueError(f"{label}: invalid coordinate: {token}")

        lon = float(values[0])
        lat = float(values[1])

        if not (
            math.isfinite(lon)
            and math.isfinite(lat)
            and -180 <= lon <= 180
            and -90 <= lat <= 90
        ):
            raise ValueError(
                f"{label}: invalid longitude/latitude: {token}"
            )

        if len(values) == 3 and not math.isfinite(float(values[2])):
            raise ValueError(f"{label}: invalid altitude: {token}")

        coordinates.append((lon, lat))

    if len(coordinates) < 4:
        raise ValueError(
            f"{label}: ring requires at least four coordinates."
        )

    first = coordinates[0]
    last = coordinates[-1]

    if not (
        math.isclose(first[0], last[0], abs_tol=1e-10)
        and math.isclose(first[1], last[1], abs_tol=1e-10)
    ):
        raise ValueError(f"{label}: ring is not closed.")

    return coordinates


def read_boundary_rings(boundary, label):
    """Read all LinearRing elements inside one boundary container."""

    linear_rings = [
        element for element in boundary.iter()
        if local_name(element.tag) == "LinearRing"
    ]

    if not linear_rings:
        raise ValueError(f"{label}: no LinearRing found.")

    return [
        read_linear_ring(
            ring,
            f"{label}, LinearRing {index}",
        )
        for index, ring in enumerate(linear_rings, start=1)
    ]


def main():
    print("=" * 65)
    print("FIRESYNC | BANDIPUR BOUNDARY EXTRACTION")
    print("=" * 65)

    if not KML_PATH.exists():
        raise FileNotFoundError(f"KML not found: {KML_PATH}")

    if OUTPUT_PATH.exists() or METADATA_PATH.exists():
        raise FileExistsError(
            "Boundary output already exists. "
            "Refusing to overwrite existing files."
        )

    tree = ET.parse(KML_PATH)
    root = tree.getroot()

    placemarks = [
        element for element in root.iter()
        if local_name(element.tag) == "Placemark"
    ]

    matches = []

    for placemark in placemarks:
        name = next(
            (
                child.text.strip()
                for child in placemark
                if local_name(child.tag) == "name"
                and child.text
            ),
            "",
        )

        if name.casefold() == "bandipur":
            matches.append(placemark)

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one Bandipur placemark; found {len(matches)}."
        )

    bandipur = matches[0]

    polygon_nodes = [
        element for element in bandipur.iter()
        if local_name(element.tag) == "Polygon"
    ]

    if not polygon_nodes:
        raise RuntimeError("No Polygon geometry found.")

    polygons = []
    ring_report = []

    print(f"\nBandipur polygon components: {len(polygon_nodes)}")

    for polygon_index, polygon_node in enumerate(
        polygon_nodes, start=1
    ):
        outer_boundaries = [
            child for child in polygon_node
            if local_name(child.tag) == "outerBoundaryIs"
        ]

        inner_boundaries = [
            child for child in polygon_node
            if local_name(child.tag) == "innerBoundaryIs"
        ]

        if len(outer_boundaries) != 1:
            raise ValueError(
                f"Polygon {polygon_index}: expected one outer boundary."
            )

        outer_rings = read_boundary_rings(
            outer_boundaries[0],
            f"Polygon {polygon_index} exterior",
        )

        if len(outer_rings) != 1:
            raise ValueError(
                f"Polygon {polygon_index}: expected one exterior ring."
            )

        shell = outer_rings[0]
        holes = []

        for boundary_index, boundary in enumerate(
            inner_boundaries, start=1
        ):
            boundary_rings = read_boundary_rings(
                boundary,
                f"Polygon {polygon_index} interior boundary {boundary_index}",
            )

            holes.extend(boundary_rings)

        geometry = Polygon(shell, holes)

        print(f"\nPolygon {polygon_index}")
        print(f"  Exterior vertices: {len(shell)}")
        print(f"  Interior rings: {len(holes)}")
        print(f"  Geometry valid: {geometry.is_valid}")

        for index, hole in enumerate(holes, start=1):
            print(
                f"  Interior ring {index} vertices: {len(hole)}"
            )

        if not geometry.is_valid:
            raise RuntimeError(
                f"Invalid polygon: {explain_validity(geometry)}"
            )

        polygons.append(geometry)

        ring_report.append({
            "polygon_component": polygon_index,
            "exterior_vertex_count": len(shell),
            "interior_ring_count": len(holes),
            "interior_vertex_counts": [
                len(hole) for hole in holes
            ],
            "valid": geometry.is_valid,
        })

    # Keep source polygon components and also construct a combined
    # geometry for overall validation and spatial operations.
    boundary = unary_union(polygons)

    if not boundary.is_valid:
        raise RuntimeError(
            "Combined boundary is invalid: "
            + explain_validity(boundary)
        )

    print("\nCOMBINED GEOMETRY")
    print(f"Geometry type: {boundary.geom_type}")
    print(f"Valid: {boundary.is_valid}")

    min_x, min_y, max_x, max_y = boundary.bounds

    print(
        "Bounds (W, S, E, N): "
        f"{min_x:.6f}, {min_y:.6f}, "
        f"{max_x:.6f}, {max_y:.6f}"
    )

    intersects_query = boundary.intersects(box(*QUERY_BBOX))

    print(f"Intersects acquisition rectangle: {intersects_query}")

    if not intersects_query:
        raise RuntimeError(
            "Bandipur boundary does not intersect the acquisition rectangle."
        )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:32643",
        always_xy=True,
    )

    projected = transform(transformer.transform, boundary)
    area_km2 = projected.area / 1_000_000

    print(f"Projected area: {area_km2:.3f} km²")

    features = []

    for index, polygon in enumerate(polygons, start=1):
        features.append({
            "type": "Feature",
            "properties": {
                "reserve": "Bandipur",
                "source_component": index,
                "source_file": KML_PATH.name,
                "boundary_status": "provisional",
            },
            "geometry": mapping(polygon),
        })

    feature_collection = {
        "type": "FeatureCollection",
        "features": features,
    }

    metadata = {
        "reserve": "Bandipur",
        "source_file": KML_PATH.name,
        "source_placemark": "Bandipur",
        "source_placemark_count": len(placemarks),
        "polygon_components": len(polygons),
        "rings": ring_report,
        "source_crs": "EPSG:4326",
        "output_crs": "EPSG:4326",
        "coordinate_order": "longitude, latitude",
        "combined_geometry_type": boundary.geom_type,
        "geometry_valid": boundary.is_valid,
        "intersects_acquisition_bbox": intersects_query,
        "bounds_west_south_east_north": [
            min_x, min_y, max_x, max_y
        ],
        "area_km2_projected_utm_43n": area_km2,
        "extracted_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": (
            "Provisional boundary. Source currency and "
            "legal boundary accuracy require independent verification."
        ),
    }

    OUTPUT_PATH.write_text(
        json.dumps(feature_collection, indent=2),
        encoding="utf-8",
    )

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nEXPORT SUCCESSFUL")
    print(f"GeoJSON: {OUTPUT_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print("Original KML remains unchanged.")
    print("=" * 65)


if __name__ == "__main__":
    main()
