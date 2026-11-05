from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json


ROOT = Path(__file__).resolve().parents[1]

SERVICE_URL = (
    "https://livingatlas.esri.in/server1/rest/services/"
    "Wildlife/IN_WildlifeProtectedAreas_2022/MapServer/0/query"
)

OUTPUT_DIR = ROOT / "data" / "boundaries" / "source"

OUTPUT_FILE = (
    OUTPUT_DIR / "esri_livingatlas_bandipur_2022.geojson"
)

METADATA_FILE = (
    OUTPUT_DIR / "esri_livingatlas_bandipur_2022.metadata.json"
)


def main():
    print("=" * 65)
    print("FIRESYNC | INDEPENDENT BOUNDARY ACQUISITION")
    print("=" * 65)

    if OUTPUT_FILE.exists() or METADATA_FILE.exists():
        raise FileExistsError(
            "Independent boundary files already exist. "
            "Refusing to overwrite."
        )

    params = {
        "where": "name LIKE '%Bandipur%' AND state LIKE '%Karnataka%'",
        "outFields": "objectid,id,name,type,state,areasqkm",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }

    url = SERVICE_URL + "?" + urlencode(params)

    request = Request(
        url,
        headers={"User-Agent": "FIRESYNC/0.1"},
    )

    try:
        with urlopen(request, timeout=60) as response:
            payload = response.read()

    except Exception as exc:
        raise RuntimeError(
            f"Independent boundary request failed: {exc}"
        ) from exc

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "The service did not return valid JSON."
        ) from exc

    if "error" in data:
        raise RuntimeError(
            f"ArcGIS service error: {data['error']}"
        )

    features = data.get("features", [])

    if not features:
        raise RuntimeError(
            "No Bandipur features returned. "
            "We need to inspect the service query."
        )

    print(f"\nFeatures returned: {len(features)}")

    for index, feature in enumerate(features, start=1):
        properties = feature.get("properties", {})
        geometry = feature.get("geometry")

        print(f"\nFEATURE {index}")
        print("Name:", properties.get("name"))
        print("Type:", properties.get("type"))
        print("State:", properties.get("state"))
        print("Reported area:", properties.get("areasqkm"))
        print(
            "Geometry:",
            geometry.get("type") if geometry else "MISSING",
        )

        if not geometry or not geometry.get("coordinates"):
            raise RuntimeError(
                f"Feature {index} has missing geometry."
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Preserve the original service response.
    OUTPUT_FILE.write_bytes(payload)

    metadata = {
        "project": "FIRESYNC",
        "source": "Esri India Living Atlas",
        "service": SERVICE_URL,
        "layer": "Wildlife Protected Areas",
        "layer_version_label": "2022",
        "query": params,
        "source_crs_requested": "EPSG:4326",
        "feature_count": len(features),
        "downloaded_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "Independent comparison source; not legally verified.",
    }

    METADATA_FILE.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nDOWNLOAD SUCCESSFUL")
    print("GeoJSON:", OUTPUT_FILE)
    print("Metadata:", METADATA_FILE)
    print("Original service response preserved.")
    print("=" * 65)


if __name__ == "__main__":
    main()
