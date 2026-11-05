from pathlib import Path
from datetime import datetime, timezone
import json
import math

import pandas as pd
from shapely.geometry import Point, shape


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"

BOUNDARY_FILE = (
    ROOT / "data" / "boundaries"
    / "bandipur_ntca_provisional.geojson"
)

BOUNDARY_METADATA = BOUNDARY_FILE.with_suffix(
    ".metadata.json"
)

OUTPUT_DIR = ROOT / "data" / "interim"

AUDIT_FILE = (
    OUTPUT_DIR
    / "firms_spatial_audit_20250301_20250305.csv"
)

CANDIDATE_FILE = (
    OUTPUT_DIR
    / "bandipur_candidate_observations_20250301_20250305.csv"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "bandipur_spatial_audit_20250301_20250305.json"
)

EXPECTED_PRODUCTS = {
    "MODIS_SP",
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_SP",
    "VIIRS_NOAA21_SP",
}


def classify_point(longitude, latitude, geometries):
    """Classify a coordinate against all boundary components."""

    if not (
        math.isfinite(longitude)
        and math.isfinite(latitude)
        and -180 <= longitude <= 180
        and -90 <= latitude <= 90
    ):
        return "invalid_coordinate"

    point = Point(longitude, latitude)

    if any(geometry.contains(point) for geometry in geometries):
        return "inside"

    if any(geometry.touches(point) for geometry in geometries):
        return "boundary_touch"

    return "outside"


def main():
    print("=" * 65)
    print("FIRESYNC | BANDIPUR POINT-IN-POLYGON ANALYSIS")
    print("=" * 65)

    required_files = [BOUNDARY_FILE, BOUNDARY_METADATA]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required boundary file missing: {path}"
            )

    with BOUNDARY_METADATA.open(
        encoding="utf-8"
    ) as file:
        boundary_metadata = json.load(file)

    if boundary_metadata.get("output_crs") != "EPSG:4326":
        raise RuntimeError(
            "Expected boundary CRS EPSG:4326."
        )

    if boundary_metadata.get("geometry_valid") is not True:
        raise RuntimeError(
            "Boundary metadata does not confirm valid geometry."
        )

    if boundary_metadata.get("status", "").find(
        "Provisional"
    ) == -1:
        raise RuntimeError(
            "Unexpected boundary provenance status."
        )

    with BOUNDARY_FILE.open(
        encoding="utf-8"
    ) as file:
        boundary_data = json.load(file)

    geometries = [
        shape(feature["geometry"])
        for feature in boundary_data["features"]
        if feature.get("geometry")
    ]

    if not geometries:
        raise RuntimeError("No boundary geometries found.")

    for index, geometry in enumerate(geometries, start=1):
        if geometry.is_empty or not geometry.is_valid:
            raise RuntimeError(
                f"Boundary geometry {index} is empty or invalid."
            )

    raw_files = sorted(RAW_DIR.glob("*/*/*.csv"))

    if not raw_files:
        raise FileNotFoundError("No NASA FIRMS CSV files found.")

    frames = []

    for path in raw_files:
        product = path.parent.name

        if product not in EXPECTED_PRODUCTS:
            continue

        print(f"\nInspecting {product}")

        data = pd.read_csv(path)

        required_columns = {
            "latitude",
            "longitude",
            "acq_date",
        }

        missing = required_columns - set(data.columns)

        if missing:
            raise RuntimeError(
                f"{product} is missing columns: {sorted(missing)}"
            )

        latitudes = pd.to_numeric(
            data["latitude"], errors="coerce"
        )

        longitudes = pd.to_numeric(
            data["longitude"], errors="coerce"
        )

        classifications = []

        for longitude, latitude in zip(
            longitudes, latitudes
        ):
            if pd.isna(longitude) or pd.isna(latitude):
                classifications.append("invalid_coordinate")
            else:
                classifications.append(
                    classify_point(
                        float(longitude),
                        float(latitude),
                        geometries,
                    )
                )

        # Preserve original NASA columns and add FIRESYNC audit fields.
        data["firms_product"] = product
        data["source_file"] = path.name
        data["boundary_classification"] = classifications
        data["inside_bandipur_strict"] = [
            value == "inside"
            for value in classifications
        ]
        data["boundary_source_status"] = "provisional"

        print(f"Records: {len(data)}")
        print(
            "Strictly inside:",
            classifications.count("inside"),
        )
        print(
            "Boundary touches:",
            classifications.count("boundary_touch"),
        )
        print(
            "Outside:",
            classifications.count("outside"),
        )
        print(
            "Invalid coordinates:",
            classifications.count("invalid_coordinate"),
        )

        frames.append(data)

    if not frames:
        raise RuntimeError(
            "No expected MODIS or VIIRS products were found."
        )

    if {frame["firms_product"].iloc[0] for frame in frames} != EXPECTED_PRODUCTS:
        raise RuntimeError(
            "Not all four expected satellite products were found."
        )

    audit = pd.concat(frames, ignore_index=True)

    candidates = audit[
        audit["inside_bandipur_strict"]
    ].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in [AUDIT_FILE, CANDIDATE_FILE, REPORT_FILE]:
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite previous result: {path}"
            )

    audit.to_csv(AUDIT_FILE, index=False)
    candidates.to_csv(CANDIDATE_FILE, index=False)

    summary = {
        "project": "FIRESYNC",
        "study_area": "Bandipur",
        "boundary_file": BOUNDARY_FILE.name,
        "boundary_status": "provisional",
        "boundary_crs": "EPSG:4326",
        "spatial_method": "Shapely Polygon.contains",
        "boundary_points_policy": (
            "Boundary-touching points are classified separately "
            "and excluded from strict interior candidates."
        ),
        "source_products": sorted(EXPECTED_PRODUCTS),
        "total_observations": int(len(audit)),
        "strictly_inside": int(
            (audit["boundary_classification"] == "inside").sum()
        ),
        "boundary_touch": int(
            (audit["boundary_classification"] == "boundary_touch").sum()
        ),
        "outside": int(
            (audit["boundary_classification"] == "outside").sum()
        ),
        "invalid_coordinates": int(
            (
                audit["boundary_classification"]
                == "invalid_coordinate"
            ).sum()
        ),
        "candidate_observations": int(len(candidates)),
        "per_product": {},
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "raw_data_modified": False,
        "outputs": {
            "audit": str(AUDIT_FILE.relative_to(ROOT)),
            "candidates": str(CANDIDATE_FILE.relative_to(ROOT)),
        },
    }

    for product in sorted(EXPECTED_PRODUCTS):
        subset = audit[
            audit["firms_product"] == product
        ]

        summary["per_product"][product] = {
            "total": int(len(subset)),
            "inside": int(
                (subset["boundary_classification"] == "inside").sum()
            ),
            "boundary_touch": int(
                (
                    subset["boundary_classification"]
                    == "boundary_touch"
                ).sum()
            ),
            "outside": int(
                (subset["boundary_classification"] == "outside").sum()
            ),
            "invalid": int(
                (
                    subset["boundary_classification"]
                    == "invalid_coordinate"
                ).sum()
            ),
        }

    REPORT_FILE.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("FINAL SPATIAL ANALYSIS")
    print("=" * 65)

    print(f"Total observations: {len(audit)}")
    print(f"Strictly inside Bandipur: {len(candidates)}")
    print(
        "Boundary touches:",
        summary["boundary_touch"],
    )
    print(f"Outside: {summary['outside']}")
    print(f"Invalid: {summary['invalid_coordinates']}")

    print("\nOUTPUT FILES")
    print(AUDIT_FILE)
    print(CANDIDATE_FILE)
    print(REPORT_FILE)

    print("\nOriginal NASA data remains unchanged.")
    print("=" * 65)


if __name__ == "__main__":
    main()
