
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

BASE_URL = "https://firms.modaps.eosdis.nasa.gov"

SUPPORTED_PRODUCTS = {
    "MODIS_SP",
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_SP",
    "VIIRS_NOAA21_SP",
}


def get_map_key() -> str:
    key = os.getenv("FIRMS_MAP_KEY")

    if not key:
        raise RuntimeError("FIRMS_MAP_KEY is missing from .env")

    return key.strip()


def _fetch_csv(url: str) -> pd.DataFrame:
    request = Request(
        url,
        headers={"User-Agent": "FIRESYNC/0.1"},
    )

    try:
        with urlopen(request, timeout=60) as response:
            content = response.read().decode("utf-8-sig")

    except HTTPError as exc:
        raise RuntimeError(
            f"NASA FIRMS returned HTTP {exc.code}"
        ) from None

    except URLError:
        raise RuntimeError(
            "Could not connect to NASA FIRMS."
        ) from None

    if not content.strip():
        raise RuntimeError("NASA returned an empty response.")

    try:
        return pd.read_csv(StringIO(content))
    except Exception:
        raise RuntimeError(
            "NASA returned an invalid CSV response."
        ) from None


def fetch_data_availability(sensor: str = "all") -> pd.DataFrame:
    sensor = sensor.strip().upper()

    allowed = {"ALL"} | SUPPORTED_PRODUCTS

    if sensor not in allowed:
        raise ValueError(f"Unsupported product: {sensor}")

    url = (
        f"{BASE_URL}/api/data_availability/csv/"
        f"{quote(get_map_key(), safe='')}/{sensor}"
    )

    data = _fetch_csv(url)

    required = {"data_id", "min_date", "max_date"}

    if not required.issubset(data.columns):
        raise RuntimeError("Unexpected availability response.")

    data["data_id"] = data["data_id"].astype(str).str.strip()
    data["min_date"] = pd.to_datetime(
        data["min_date"], errors="coerce"
    )
    data["max_date"] = pd.to_datetime(
        data["max_date"], errors="coerce"
    )

    return data


def download_area_csv(
    source: str,
    west: float,
    south: float,
    east: float,
    north: float,
    start_date: str,
    day_range: int = 5,
) -> tuple[Path, pd.DataFrame]:
    """Download and preserve original NASA FIRMS CSV observations."""

    source = source.strip().upper()

    if source not in SUPPORTED_PRODUCTS:
        raise ValueError(f"Unsupported product: {source}")

    if not 1 <= day_range <= 5:
        raise ValueError("FIRMS day_range must be between 1 and 5.")

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = start + timedelta(days=day_range - 1)

    if not (
        -180 <= west < east <= 180
        and -90 <= south < north <= 90
    ):
        raise ValueError("Invalid bounding box coordinates.")

    area = f"{west},{south},{east},{north}"
    key = quote(get_map_key(), safe="")

    url = (
        f"{BASE_URL}/api/area/csv/"
        f"{key}/{source}/{area}/{day_range}/{start_date}"
    )

    request = Request(
        url,
        headers={"User-Agent": "FIRESYNC/0.1"},
    )

    try:
        with urlopen(request, timeout=90) as response:
            payload = response.read()

    except HTTPError as exc:
        raise RuntimeError(
            f"NASA FIRMS returned HTTP {exc.code} for {source}."
        ) from None

    except URLError:
        raise RuntimeError(
            f"Network connection failed for {source}."
        ) from None

    try:
        data = pd.read_csv(
            StringIO(payload.decode("utf-8-sig"))
        )
    except Exception:
        raise RuntimeError(
            f"NASA returned an invalid CSV for {source}."
        ) from None

    required = {"latitude", "longitude", "acq_date"}

    if not required.issubset(data.columns):
        raise RuntimeError(
            f"NASA response for {source} lacks required columns."
        )

    folder = (
        "modis"
        if source.startswith("MODIS")
        else "viirs"
    )

    output_dir = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / folder
        / source
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{source}_{start.isoformat()}_to_{end.isoformat()}.csv"
    )

    output_path = output_dir / filename
    metadata_path = output_path.with_suffix(".json")

    if output_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing raw data: {filename}"
        )

    # Preserve the original NASA CSV bytes.
    output_path.write_bytes(payload)

    metadata = {
        "source": source,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "day_range": day_range,
        "bounding_box_west_south_east_north": [
            west, south, east, north
        ],
        "record_count": len(data),
        "downloaded_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "original_filename": filename,
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    return output_path, data
