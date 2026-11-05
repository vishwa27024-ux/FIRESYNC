from pathlib import Path
from datetime import datetime, timezone
import json
import warnings
import xml.etree.ElementTree as ET

import pandas as pd
from pyproj import Geod, Transformer
from shapely.geometry import Point, shape
from shapely.ops import transform


ROOT = Path(__file__).resolve().parents[1]

NTCA_GEOJSON = (
    ROOT / "data" / "boundaries"
    / "bandipur_ntca_provisional.geojson"
)

ESRI_GEOJSON = (
    ROOT / "data" / "boundaries" / "source"
    / "esri_livingatlas_bandipur_2022.geojson"
)

NTCA_KML = (
    ROOT / "data" / "boundaries" / "source"
    / "ntca_extracted" / "PA_TR_Corridor_Final.kml"
)

QC_FILE = (
    ROOT / "data" / "interim"
    / "spatial_quality_control.csv"
)

OUTPUT_DIR = ROOT / "data" / "interim"

REPORT_FILE = OUTPUT_DIR / "boundary_geometry_comparison.json"
POINTS_FILE = OUTPUT_DIR / "boundary_point_comparison.csv"

GEOD = Geod(ellps="WGS84")

TRANSFORMER = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:32643",
    always_xy=True,
)


def load_geojson_geometry(path):
    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    features = data.get("features", [])

    if len(features) != 1:
        raise RuntimeError(
            f"{path.name}: expected one feature, found {len(features)}."
        )

    geometry_data = features[0].get("geometry")

    if not geometry_data:
        raise RuntimeError(f"{path.name}: geometry is missing.")

    geometry = shape(geometry_data)

    if geometry.is_empty or not geometry.is_valid:
        raise RuntimeError(
            f"{path.name}: geometry is empty or invalid."
        )

    return geometry, features[0].get("properties", {})



def read_ntca_reported_area():
    """Read REP_AREA from KML Data or SimpleData attributes."""

    tree = ET.parse(NTCA_KML)
    matches = []

    for placemark in tree.getroot().iter():
        if placemark.tag.split("}")[-1] != "Placemark":
            continue

        name = next(
            (
                child.text.strip()
                for child in placemark
                if child.tag.split("}")[-1] == "name"
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

    for element in matches[0].iter():
        tag = element.tag.split("}")[-1]

        if element.attrib.get("name") != "REP_AREA":
            continue

        if tag == "SimpleData":
            candidate = (element.text or "").strip()

        elif tag == "Data":
            candidate = next(
                (
                    (child.text or "").strip()
                    for child in element.iter()
                    if child.tag.split("}")[-1] == "value"
                    and child.text
                    and child.text.strip()
                ),
                "",
            )
        else:
            continue

        try:
            return float(candidate)
        except (ValueError, TypeError):
            raise RuntimeError(
                "REP_AREA value is not numeric."
            ) from None

    raise RuntimeError("REP_AREA attribute not found.")



def geodesic_area_km2(geometry):
    area, _ = GEOD.geometry_area_perimeter(geometry)
    return abs(area) / 1_000_000


def project_geometry(geometry):
    # Suppress only the Shapely legacy transform deprecation warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
        )
        return transform(TRANSFORMER.transform, geometry)


def polygon_count(geometry):
    if geometry.geom_type == "Polygon":
        return 1

    if geometry.geom_type == "MultiPolygon":
        return len(geometry.geoms)

    return 0


def hole_count(geometry):
    if geometry.geom_type == "Polygon":
        return len(geometry.interiors)

    if geometry.geom_type == "MultiPolygon":
        return sum(len(polygon.interiors) for polygon in geometry.geoms)

    return 0


def classify(point, geometry):
    if geometry.contains(point):
        return "inside"

    if geometry.touches(point):
        return "boundary_touch"

    return "outside"


def main():
    print("=" * 70)
    print("FIRESYNC | NTCA VS ESRI BOUNDARY COMPARISON")
    print("=" * 70)

    for path in [NTCA_GEOJSON, ESRI_GEOJSON, NTCA_KML, QC_FILE]:
        if not path.exists():
            raise FileNotFoundError(path)

    if REPORT_FILE.exists() or POINTS_FILE.exists():
        raise FileExistsError(
            "Comparison outputs already exist. Refusing to overwrite."
        )

    ntca, ntca_properties = load_geojson_geometry(NTCA_GEOJSON)
    esri, esri_properties = load_geojson_geometry(ESRI_GEOJSON)

    ntca_area_attribute = read_ntca_reported_area()
    esri_area_attribute = esri_properties.get("areasqkm")

    if esri_area_attribute is None:
        raise RuntimeError("Esri areasqkm attribute is missing.")

    esri_area_attribute = float(esri_area_attribute)

    print("\nSOURCE ATTRIBUTES")
    print("NTCA reported area:", ntca_area_attribute, "km²")
    print("Esri reported area:", esri_area_attribute, "km²")
    print("Esri name:", esri_properties.get("name"))
    print("Esri type:", esri_properties.get("type"))
    print("Esri state:", esri_properties.get("state"))

    print("\nGEOMETRY INSPECTION")
    print("NTCA type:", ntca.geom_type)
    print("Esri type:", esri.geom_type)
    print("NTCA valid:", ntca.is_valid)
    print("Esri valid:", esri.is_valid)
    print("NTCA bounds:", ntca.bounds)
    print("Esri bounds:", esri.bounds)
    print("NTCA polygon components:", polygon_count(ntca))
    print("Esri polygon components:", polygon_count(esri))
    print("NTCA holes:", hole_count(ntca))
    print("Esri holes:", hole_count(esri))

    # Compare topology, coordinate equality and geometric overlap.
    intersection = ntca.intersection(esri)
    union = ntca.union(esri)
    symmetric_difference = ntca.symmetric_difference(esri)

    ntca_area = geodesic_area_km2(ntca)
    esri_area = geodesic_area_km2(esri)
    intersection_area = geodesic_area_km2(intersection)
    union_area = geodesic_area_km2(union)
    difference_area = geodesic_area_km2(symmetric_difference)

    overlap_pct = (
        intersection_area / union_area * 100
        if union_area else 0
    )

    projected_ntca = project_geometry(ntca)
    projected_esri = project_geometry(esri)

    hausdorff_m = (
        projected_ntca.boundary.hausdorff_distance(
            projected_esri.boundary
        )
    )

    exact_topological_match = ntca.equals(esri)
    exact_coordinate_match = ntca.equals_exact(
        esri,
        tolerance=1e-8,
    )

    print("\nGEOMETRY COMPARISON")
    print("Topologically equal:", exact_topological_match)
    print("Coordinate-equal within 1e-8 degrees:", exact_coordinate_match)
    print(f"NTCA geodesic area: {ntca_area:.4f} km²")
    print(f"Esri geodesic area: {esri_area:.4f} km²")
    print(f"Intersection area: {intersection_area:.4f} km²")
    print(f"Union area: {union_area:.4f} km²")
    print(f"Symmetric difference: {difference_area:.4f} km²")
    print(f"Intersection / union overlap: {overlap_pct:.4f}%")
    print(f"Boundary Hausdorff distance: {hausdorff_m:.3f} metres")

    # Compare all existing observations under both boundaries.
    observations = pd.read_csv(QC_FILE)
    point_results = []

    for _, row in observations.iterrows():
        point = Point(
            float(row["longitude"]),
            float(row["latitude"]),
        )

        ntca_class = classify(point, ntca)
        esri_class = classify(point, esri)

        point_results.append({
            "product": row["product"],
            "source_file": row["source_file"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "ntca_classification": ntca_class,
            "esri_classification": esri_class,
            "classification_changed": ntca_class != esri_class,
        })

    point_report = pd.DataFrame(point_results)

    print("\nOBSERVATION CLASSIFICATION COMPARISON")
    print("-" * 70)

    print(
        pd.crosstab(
            point_report["ntca_classification"],
            point_report["esri_classification"],
        ).to_string()
    )

    print(
        "\nObservations with changed classification:",
        int(point_report["classification_changed"].sum()),
    )

    report = {
        "project": "FIRESYNC",
        "comparison_date_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "ntca_source": NTCA_GEOJSON.name,
        "esri_source": ESRI_GEOJSON.name,
        "crs": "EPSG:4326",
        "ntca_reported_area_km2": ntca_area_attribute,
        "esri_reported_area_km2": esri_area_attribute,
        "ntca_geometry_type": ntca.geom_type,
        "esri_geometry_type": esri.geom_type,
        "ntca_valid": ntca.is_valid,
        "esri_valid": esri.is_valid,
        "topologically_equal": exact_topological_match,
        "coordinate_equal_within_1e_8_degrees": exact_coordinate_match,
        "ntca_geodesic_area_km2": ntca_area,
        "esri_geodesic_area_km2": esri_area,
        "intersection_area_km2": intersection_area,
        "union_area_km2": union_area,
        "symmetric_difference_area_km2": difference_area,
        "intersection_over_union_percent": overlap_pct,
        "boundary_hausdorff_distance_m": hausdorff_m,
        "observations_compared": len(point_report),
        "observations_with_changed_classification": int(
            point_report["classification_changed"].sum()
        ),
        "boundary_independence_established": False,
        "interpretation": (
            "Geometric agreement does not independently establish "
            "source lineage or legal boundary authority."
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    REPORT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    point_report.to_csv(POINTS_FILE, index=False)

    print("\nOUTPUTS")
    print(REPORT_FILE)
    print(POINTS_FILE)

    print("\nOriginal geometries and NASA data remain unchanged.")
    print("=" * 70)


if __name__ == "__main__":
    main()
