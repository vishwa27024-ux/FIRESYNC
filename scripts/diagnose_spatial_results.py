from pathlib import Path
import warnings

import pandas as pd
from dotenv import load_dotenv
from pyproj import Transformer
from shapely.geometry import Point, Polygon, shape
from shapely.ops import transform as project_geometry


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
BOUNDARY_FILE = (
    ROOT / "data" / "boundaries"
    / "bandipur_ntca_provisional.geojson"
)

OUTPUT_FILE = (
    ROOT / "data" / "interim"
    / "spatial_quality_control.csv"
)

PRODUCTS = {
    "MODIS_SP",
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_SP",
    "VIIRS_NOAA21_SP",
}


def main():
    print("=" * 70)
    print("FIRESYNC | INDEPENDENT SPATIAL QUALITY CONTROL")
    print("=" * 70)

    if not BOUNDARY_FILE.exists():
        raise FileNotFoundError(BOUNDARY_FILE)

    import json

    with BOUNDARY_FILE.open(encoding="utf-8") as file:
        boundary_data = json.load(file)

    geometries = [
        shape(feature["geometry"])
        for feature in boundary_data["features"]
        if feature.get("geometry")
    ]

    if not geometries:
        raise RuntimeError("No boundary geometry found.")

    for geometry in geometries:
        if not geometry.is_valid:
            raise RuntimeError("Invalid boundary geometry.")

    # Independent sanity check: a representative point
    # should be strictly inside each valid polygon.
    print("\nBOUNDARY SANITY CHECK")

    for index, geometry in enumerate(geometries, start=1):
        representative = geometry.representative_point()

        print(f"\nGeometry {index}")
        print(f"Type: {geometry.geom_type}")
        print(f"Valid: {geometry.is_valid}")
        print(
            "Representative point:",
            representative.x,
            representative.y,
        )
        print(
            "Representative point inside:",
            geometry.contains(representative),
        )

        if not geometry.contains(representative):
            raise RuntimeError(
                "Boundary representative-point test failed."
            )

    # Construct exterior-only polygons to detect points
    # that may fall inside a hole in the boundary.
    exterior_polygons = [
        Polygon(geometry.exterior)
        for geometry in geometries
        if geometry.geom_type == "Polygon"
    ]

    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:32643",
        always_xy=True,
    )

    # Suppress only the known Shapely legacy transform warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
        )

        projected_boundaries = [
            project_geometry(transformer.transform, geometry)
            for geometry in geometries
        ]

    projected_edges = [
        geometry.boundary
        for geometry in projected_boundaries
    ]

    results = []

    raw_files = sorted(RAW_DIR.glob("*/*/*.csv"))

    for path in raw_files:
        product = path.parent.name

        if product not in PRODUCTS:
            continue

        data = pd.read_csv(path)

        for row_index, row in data.iterrows():
            latitude = pd.to_numeric(
                row["latitude"], errors="coerce"
            )
            longitude = pd.to_numeric(
                row["longitude"], errors="coerce"
            )

            if pd.isna(latitude) or pd.isna(longitude):
                continue

            point = Point(float(longitude), float(latitude))

            inside = any(
                geometry.contains(point)
                for geometry in geometries
            )

            touches = any(
                geometry.touches(point)
                for geometry in geometries
            )

            in_exterior = any(
                exterior.contains(point)
                for exterior in exterior_polygons
            )

            if inside:
                classification = "inside_polygon"
            elif touches:
                classification = "touching_boundary"
            elif in_exterior:
                classification = "inside_polygon_hole"
            else:
                classification = "outside_exterior"

            projected_point = project_geometry(
                transformer.transform,
                point,
            )

            distances_km = [
                edge.distance(projected_point) / 1000
                for edge in projected_edges
            ]

            nearest_distance_km = min(distances_km)

            result = {
                "product": product,
                "source_file": path.name,
                "source_row": int(row_index),
                "latitude": float(latitude),
                "longitude": float(longitude),
                "acq_date": str(row["acq_date"]),
                "classification": classification,
                "inside_outer_ring": bool(in_exterior),
                "distance_to_boundary_km": round(
                    nearest_distance_km, 3
                ),
            }

            results.append(result)

    if not results:
        raise RuntimeError("No observations were inspected.")

    report = pd.DataFrame(results)

    print("\nOBSERVATION-BY-OBSERVATION RESULTS")
    print("-" * 70)

    print(
        report[
            [
                "product",
                "latitude",
                "longitude",
                "classification",
                "distance_to_boundary_km",
            ]
        ].to_string(index=False)
    )

    print("\nCLASSIFICATION SUMMARY")
    print("-" * 70)

    print(
        report.groupby(
            ["product", "classification"]
        ).size().to_string()
    )

    print("\nDISTANCE SUMMARY (KM)")
    print("-" * 70)

    print(
        report.groupby("product")[
            "distance_to_boundary_km"
        ].agg(["min", "max", "mean"]).to_string()
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Refusing to overwrite {OUTPUT_FILE}"
        )

    report.to_csv(OUTPUT_FILE, index=False)

    print("\nOUTPUT:", OUTPUT_FILE)
    print("Original raw observations and boundary unchanged.")
    print("=" * 70)


if __name__ == "__main__":
    main()
