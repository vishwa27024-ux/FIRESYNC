"""Controlled, resumable acquisition for FIRESYNC historical FIRMS data.

Default mode is a dry run. To make network requests, pass both --execute
and --max-requests. Each completed response is validated and recorded before
the next request is considered. Ambiguous failures stop the batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

from src.data.firms_client import download_area_csv, get_map_key


ROOT = Path(__file__).resolve().parents[1]

PLAN_FILE = (
    ROOT / "data" / "interim"
    / "historical_2021_2025_plan.csv"
)

MANIFEST_FILE = (
    ROOT / "data" / "interim"
    / "historical_2021_2025_download_manifest.json"
)

AREA = {
    "west": 76.15,
    "south": 11.55,
    "east": 76.91,
    "north": 11.96,
}

BBOX = [
    AREA["west"],
    AREA["south"],
    AREA["east"],
    AREA["north"],
]

TRANSACTION_RESERVE = 500
USER_AGENT = "FIRESYNC/0.1"

SUPPORTED_SOURCES = [
    "MODIS_SP",
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_SP",
    "VIIRS_NOAA21_SP",
]

REQUIRED_PLAN_COLUMNS = {
    "source",
    "start",
    "end",
    "day_range",
    "status",
}

PLAN_TRACKING_COLUMNS = [
    "last_error",
    "transactions_before",
    "transactions_after",
    "transaction_delta",
    "quota_check",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def transaction_status() -> dict:
    """Read current NASA MAP_KEY usage without making a data request."""
    url = (
        "https://firms.modaps.eosdis.nasa.gov/"
        "mapserver/mapkey_status/?MAP_KEY="
        + quote(get_map_key(), safe="")
    )
    request = Request(url, headers={"User-Agent": USER_AGENT})

    with urlopen(request, timeout=90) as response:
        result = json.loads(response.read().decode("utf-8"))

    required = {
        "transaction_limit",
        "current_transactions",
        "transaction_interval",
    }
    if not isinstance(result, dict) or not required.issubset(result):
        raise RuntimeError("NASA returned an incomplete transaction status.")

    limit = int(result["transaction_limit"])
    current = int(result["current_transactions"])
    interval = str(result["transaction_interval"])

    if limit <= 0 or current < 0 or current > limit:
        raise RuntimeError("NASA returned invalid transaction counters.")

    return {
        "limit": limit,
        "current": current,
        "interval": interval,
    }


def get_paths(source: str, start: str, end: str) -> tuple[Path, Path]:
    folder = "modis" if source.startswith("MODIS") else "viirs"
    filename = f"{source}_{start}_to_{end}.csv"
    csv_path = ROOT / "data" / "raw" / folder / source / filename
    return csv_path, csv_path.with_suffix(".json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_window(source: str, window: dict) -> None:
    if source not in SUPPORTED_SOURCES:
        raise ValueError(f"Unsupported source: {source}")

    start = date.fromisoformat(str(window["start"]))
    end = date.fromisoformat(str(window["end"]))
    day_range = int(window["day_range"])

    if not 1 <= day_range <= 5:
        raise ValueError("FIRMS day_range must be between 1 and 5.")

    if (end - start).days + 1 != day_range:
        raise ValueError("Date window and day_range do not agree.")


def validate_saved_chunk(source: str, window: dict) -> dict:
    """Validate original CSV bytes and sidecar metadata, including empty CSVs."""
    validate_window(source, window)
    csv_path, metadata_path = get_paths(
        source,
        window["start"],
        window["end"],
    )

    if not csv_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("CSV and metadata must both exist.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_metadata = {
        "source": source,
        "start_date": window["start"],
        "end_date": window["end"],
        "day_range": int(window["day_range"]),
    }

    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Metadata mismatch: {key}")

    actual_bbox = metadata.get("bounding_box_west_south_east_north")
    if (
        not isinstance(actual_bbox, list)
        or len(actual_bbox) != 4
        or not all(
            math.isclose(float(actual), float(expected), abs_tol=1e-8)
            for actual, expected in zip(actual_bbox, BBOX)
        )
    ):
        raise ValueError("Bounding box metadata mismatch.")

    data = pd.read_csv(csv_path)
    required_csv_columns = {"latitude", "longitude", "acq_date"}
    if not required_csv_columns.issubset(data.columns):
        raise ValueError("Required CSV columns are missing.")

    if len(data) != int(metadata.get("record_count", -1)):
        raise ValueError("CSV record count does not match metadata.")

    if not data.empty:
        latitude = pd.to_numeric(data["latitude"], errors="coerce")
        longitude = pd.to_numeric(data["longitude"], errors="coerce")
        acquisition_dates = pd.to_datetime(data["acq_date"], errors="coerce")

        if latitude.isna().any() or longitude.isna().any():
            raise ValueError("Invalid coordinates detected.")
        if not latitude.between(-90, 90).all():
            raise ValueError("Latitude outside valid range.")
        if not longitude.between(-180, 180).all():
            raise ValueError("Longitude outside valid range.")
        if acquisition_dates.isna().any():
            raise ValueError("Invalid acquisition date detected.")
        if not acquisition_dates.between(
            pd.Timestamp(window["start"]),
            pd.Timestamp(window["end"]),
        ).all():
            raise ValueError("Observation outside requested date window.")

    return {
        "record_count": len(data),
        "file_bytes": csv_path.stat().st_size,
        "sha256": sha256_file(csv_path),
        "csv_file": csv_path.relative_to(ROOT).as_posix(),
        "metadata_file": metadata_path.relative_to(ROOT).as_posix(),
    }


def save_plan(full_plan: pd.DataFrame) -> None:
    """Atomically save the complete plan, never a filtered batch subset."""
    PLAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = PLAN_FILE.with_suffix(".tmp")
    full_plan.to_csv(temporary, index=False)
    temporary.replace(PLAN_FILE)


def load_manifest() -> dict:
    if not MANIFEST_FILE.exists():
        return {
            "project": "FIRESYNC",
            "period": "2021-01-01 to 2025-12-31",
            "bounding_box": AREA,
            "attempts": [],
        }

    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("attempts"), list
    ):
        raise ValueError("Existing historical download manifest is invalid.")

    if manifest.get("project") != "FIRESYNC":
        raise ValueError("Manifest belongs to a different project.")
    if manifest.get("bounding_box") != AREA:
        raise ValueError("Manifest bounding box does not match this project.")

    return manifest


def save_manifest(manifest: dict) -> None:
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at_utc"] = utc_now()
    temporary = MANIFEST_FILE.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    temporary.replace(MANIFEST_FILE)


def load_full_plan() -> pd.DataFrame:
    if not PLAN_FILE.is_file():
        raise FileNotFoundError(
            f"Historical plan not found: {PLAN_FILE}. Run the planner first."
        )

    full_plan = pd.read_csv(PLAN_FILE, dtype={
        "source": "string",
        "start": "string",
        "end": "string",
        "status": "string",
    })

    missing_columns = REQUIRED_PLAN_COLUMNS - set(full_plan.columns)
    if missing_columns:
        raise ValueError(
            "Historical plan is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if full_plan.empty:
        raise ValueError("Historical plan contains no rows.")

    if full_plan.duplicated(["source", "start", "end"]).any():
        raise ValueError("Historical plan contains duplicate source/date windows.")

    invalid_sources = set(full_plan["source"].dropna()) - set(SUPPORTED_SOURCES)
    if invalid_sources:
        raise ValueError(
            "Historical plan contains unsupported sources: "
            + ", ".join(sorted(invalid_sources))
        )

    # Add audit columns to the FULL inventory before any selection is made.
    # Explicit dtypes avoid pandas warnings when hashes/errors are written
    # into columns that are initially empty in the planner output.
    text_columns = ["last_error", "quota_check", "sha256"]
    integer_columns = [
        "transactions_before",
        "transactions_after",
        "transaction_delta",
        "record_count",
        "file_bytes",
    ]

    for column in text_columns:
        if column not in full_plan.columns:
            full_plan[column] = pd.Series(
                pd.NA, index=full_plan.index, dtype="string"
            )
        else:
            full_plan[column] = full_plan[column].astype("string")

    for column in integer_columns:
        if column not in full_plan.columns:
            full_plan[column] = pd.Series(
                pd.NA, index=full_plan.index, dtype="Int64"
            )
        else:
            full_plan[column] = pd.to_numeric(
                full_plan[column], errors="coerce"
            ).astype("Int64")

    return full_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled FIRESYNC historical FIRMS acquisition."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Issue NASA data requests. Default behaviour is dry-run only.",
    )
    parser.add_argument(
        "--start-date",
        help="Select windows beginning on this ISO date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=SUPPORTED_SOURCES,
        help="Restrict acquisition to selected FIRMS products.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        help="Maximum number of data requests in this run.",
    )

    args = parser.parse_args()

    if args.execute and (args.max_requests is None or args.max_requests < 1):
        parser.error("--execute requires a positive --max-requests limit.")
    if args.max_requests is not None and args.max_requests < 1:
        parser.error("--max-requests must be positive.")
    if args.start_date:
        try:
            parsed = date.fromisoformat(args.start_date)
        except ValueError:
            parser.error("--start-date must use YYYY-MM-DD format.")
        if parsed.isoformat() != args.start_date:
            parser.error("--start-date must use YYYY-MM-DD format.")

    return args


def main() -> None:
    args = parse_args()
    full_plan = load_full_plan()

    # Selection is separate from the full inventory. All writes below update
    # full_plan by its original row index and then save all 1,409 rows.
    selection = full_plan["status"].eq("missing_download")

    if args.sources:
        selection &= full_plan["source"].isin(args.sources)
    if args.start_date:
        selection &= full_plan["start"].astype(str).eq(args.start_date)

    selected = full_plan.loc[selection].sort_values(
        ["source", "start"]
    ).copy()

    if args.max_requests is not None:
        selected = selected.head(args.max_requests).copy()

    print("=" * 72)
    print("FIRESYNC | CONTROLLED HISTORICAL DOWNLOAD")
    print("=" * 72)
    print(f"Full plan rows: {len(full_plan)}")
    print(f"Selected requests: {len(selected)}")
    print(f"Execution enabled: {args.execute}")

    for _, row in selected.iterrows():
        print(
            f"{row['source']} | {row['start']} to {row['end']} | "
            f"{int(row['day_range'])} days"
        )

    if selected.empty:
        print("No missing windows match the selected criteria.")
        return

    if not args.execute:
        print("\nDRY RUN ONLY. No NASA requests were made.")
        print("Add --execute and --max-requests to authorize requests.")
        return

    manifest = load_manifest()
    request_count = 0
    stop_reason = None

    for index, row in selected.iterrows():
        source = str(row["source"])
        window = {
            "start": str(row["start"]),
            "end": str(row["end"]),
            "day_range": int(row["day_range"]),
        }
        validate_window(source, window)

        csv_path, metadata_path = get_paths(
            source, window["start"], window["end"]
        )
        print(f"\nProcessing {source} | {window['start']} to {window['end']}")

        # Re-check disk immediately before a request. Never overwrite raw data.
        if csv_path.exists() or metadata_path.exists():
            if csv_path.exists() and metadata_path.exists():
                try:
                    result = validate_saved_chunk(source, window)
                    existing_status = (
                        "existing_valid_empty"
                        if result["record_count"] == 0
                        else "existing_valid"
                    )
                    full_plan.at[index, "status"] = existing_status
                    full_plan.at[index, "record_count"] = result["record_count"]
                    full_plan.at[index, "file_bytes"] = result["file_bytes"]
                    full_plan.at[index, "sha256"] = result["sha256"]
                    full_plan.at[index, "last_error"] = pd.NA
                    print(f"Existing files validated: {existing_status}")
                except Exception as error:
                    full_plan.at[index, "status"] = "invalid_existing"
                    full_plan.at[index, "last_error"] = str(error)
                    print(f"Existing files need manual inspection: {error}")
            else:
                full_plan.at[index, "status"] = "incomplete_pair"
                full_plan.at[index, "last_error"] = (
                    "Only CSV or metadata exists; refusing overwrite."
                )
                print("Incomplete CSV/metadata pair. Refusing overwrite.")

            save_plan(full_plan)
            continue

        # Quota status must be known before every data request.
        try:
            before = transaction_status()
        except Exception as error:
            full_plan.at[index, "last_error"] = (
                f"Pre-request quota check failed; no data request issued: {error}"
            )
            save_plan(full_plan)
            stop_reason = "pre-request quota status unavailable"
            print(stop_reason + ". Stopping without issuing a data request.")
            break

        available = before["limit"] - before["current"]
        print(
            f"Transactions: {before['current']}/{before['limit']} "
            f"({before['interval']})"
        )

        if available <= TRANSACTION_RESERVE:
            full_plan.at[index, "last_error"] = (
                f"Safety reserve reached: {available} transactions remain."
            )
            full_plan.at[index, "quota_check"] = "safety_reserve_reached"
            save_plan(full_plan)
            stop_reason = "transaction safety reserve reached"
            print("Safety reserve reached. Stopping before this request.")
            break

        full_plan.at[index, "transactions_before"] = before["current"]

        # Do not auto-retry. A client timeout can occur after NASA processes
        # the request, so an automatic retry could create a duplicate request.
        try:
            download_area_csv(
                source=source,
                start_date=window["start"],
                day_range=window["day_range"],
                **AREA,
            )
        except Exception as error:
            full_plan.at[index, "status"] = "request_failed_manual_review"
            full_plan.at[index, "last_error"] = str(error)
            full_plan.at[index, "quota_check"] = "request_outcome_ambiguous"

            manifest["attempts"].append({
                "source": source,
                **window,
                "status": "request_failed_manual_review",
                "error": str(error),
                "quota_check": "request_outcome_ambiguous",
                "timestamp_utc": utc_now(),
            })
            save_plan(full_plan)
            save_manifest(manifest)
            stop_reason = "request outcome ambiguous"
            print(f"Request error: {error}")
            print("Stopping to avoid an unsafe automatic retry.")
            break

        # A returned request must have a valid CSV+metadata pair.
        try:
            result = validate_saved_chunk(source, window)
        except Exception as error:
            full_plan.at[index, "status"] = "downloaded_invalid"
            full_plan.at[index, "last_error"] = str(error)
            full_plan.at[index, "quota_check"] = "not_checked_after_invalid_response"
            manifest["attempts"].append({
                "source": source,
                **window,
                "status": "downloaded_invalid",
                "error": str(error),
                "timestamp_utc": utc_now(),
            })
            save_plan(full_plan)
            save_manifest(manifest)
            stop_reason = "downloaded response failed validation"
            print(f"Downloaded response failed validation: {error}")
            print("Stopping the batch for manual review.")
            break

        download_status = (
            "downloaded_empty"
            if result["record_count"] == 0
            else "downloaded_valid"
        )
        full_plan.at[index, "status"] = download_status
        full_plan.at[index, "record_count"] = result["record_count"]
        full_plan.at[index, "file_bytes"] = result["file_bytes"]
        full_plan.at[index, "sha256"] = result["sha256"]
        full_plan.at[index, "last_error"] = pd.NA

        # The counter delta is an observed change. It can be attributed to
        # this request only if no other process uses the same MAP_KEY.
        quota_check = "post_request_status_unavailable"
        delta = None
        after = None
        post_error = None

        try:
            after = transaction_status()
            if (
                after["limit"] != before["limit"]
                or after["interval"] != before["interval"]
            ):
                quota_check = "counter_configuration_changed"
            elif after["current"] > before["current"]:
                delta = after["current"] - before["current"]
                quota_check = "counter_increased_exclusive_key_use_assumed"
            elif after["current"] == before["current"]:
                quota_check = "counter_unchanged"
            else:
                quota_check = "counter_decreased_or_interval_reset"
        except Exception as error:
            post_error = str(error)

        full_plan.at[index, "transactions_after"] = (
            after["current"] if after is not None else pd.NA
        )
        full_plan.at[index, "transaction_delta"] = (
            delta if delta is not None else pd.NA
        )
        full_plan.at[index, "quota_check"] = quota_check
        if post_error:
            full_plan.at[index, "last_error"] = (
                "Data response saved and validated; post-request quota check failed: "
                + post_error
            )

        manifest["attempts"].append({
            "source": source,
            **window,
            "status": download_status,
            "record_count": result["record_count"],
            "sha256": result["sha256"],
            "transactions_before": before["current"],
            "transactions_after": (
                after["current"] if after is not None else None
            ),
            "transaction_delta_observed": delta,
            "quota_check": quota_check,
            "quota_attribution_note": (
                "Observed counter increase; attribution to this request assumes "
                "no concurrent use of the same MAP_KEY."
            ),
            "post_request_status_error": post_error,
            "timestamp_utc": utc_now(),
        })

        # Crucial: write the entire 1,409-row plan, not the selected subset.
        save_plan(full_plan)
        save_manifest(manifest)

        request_count += 1
        print(f"Saved and validated: {download_status}")
        print(f"Records: {result['record_count']}")
        print(f"SHA-256: {result['sha256']}")
        print(f"Quota check: {quota_check}")

        if quota_check != "counter_increased_exclusive_key_use_assumed":
            stop_reason = f"quota verification not conclusive: {quota_check}"
            print("Stopping because transaction usage could not be verified safely.")
            break

    print("\n" + "=" * 72)
    print("BATCH FINISHED")
    print("=" * 72)
    print(f"Validated requests in this run: {request_count}")
    if stop_reason:
        print(f"Stop reason: {stop_reason}")
    print(f"Full plan: {PLAN_FILE}")
    print(f"Manifest: {MANIFEST_FILE}")
    print(f"Full plan rows preserved: {len(full_plan)}")


if __name__ == "__main__":
    main()
