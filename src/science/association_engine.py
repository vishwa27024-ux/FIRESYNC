
"""Explainable candidate association engine for FIRESYNC.

This module links MODIS and VIIRS observations using configurable
spatial and temporal thresholds. Links are candidates, not confirmed
same-fire events.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from pyproj import Geod


REQUIRED_COLUMNS = {
    "observation_id",
    "sensor_family",
    "satellite_platform",
    "latitude",
    "longitude",
    "acq_datetime_utc",
}

_GEOD = Geod(ellps="WGS84")


@dataclass
class AssociationResult:
    """Outputs from one association run."""

    edges: pd.DataFrame
    clusters: pd.DataFrame
    unmatched: pd.DataFrame


def _validate_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Validate and prepare observations without modifying the input."""

    missing = REQUIRED_COLUMNS.difference(observations.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = observations.copy()

    if data["observation_id"].isna().any():
        raise ValueError("observation_id contains null values.")

    if data["observation_id"].duplicated().any():
        raise ValueError("observation_id must be unique.")

    data["sensor_family"] = (
        data["sensor_family"].astype("string").str.upper()
    )

    data["latitude"] = pd.to_numeric(data["latitude"], errors="coerce")
    data["longitude"] = pd.to_numeric(data["longitude"], errors="coerce")
    data["acq_datetime_utc"] = pd.to_datetime(
        data["acq_datetime_utc"], utc=True, errors="coerce"
    )

    if data[["latitude", "longitude"]].isna().any().any():
        raise ValueError("Latitude or longitude contains invalid values.")

    if data["acq_datetime_utc"].isna().any():
        raise ValueError("Invalid or missing acquisition timestamps.")

    if not data["latitude"].between(-90, 90).all():
        raise ValueError("Latitude must be between -90 and 90.")

    if not data["longitude"].between(-180, 180).all():
        raise ValueError("Longitude must be between -180 and 180.")

    unexpected = set(data["sensor_family"].dropna().unique()) - {
        "MODIS", "VIIRS"
    }
    if unexpected:
        raise ValueError(f"Unexpected sensor families: {sorted(unexpected)}")

    return data.reset_index(drop=True)


def _distance_km(lon1, lat1, lon2, lat2) -> float:
    """Calculate WGS84 ellipsoidal distance in kilometres."""

    _, _, distance_m = _GEOD.inv(lon1, lat1, lon2, lat2)
    return float(distance_m / 1000.0)


class _UnionFind:
    """Small disjoint-set structure for connected components."""

    def __init__(self, values):
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value):
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first, second):
        root_a = self.find(first)
        root_b = self.find(second)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a

        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def build_associations(
    observations: pd.DataFrame,
    distance_threshold_km: float = 1.0,
    time_window_minutes: float = 60.0,
) -> AssociationResult:
    """Build MODIS–VIIRS candidate edges and connected clusters.

    A candidate edge exists only when:
      1. One observation is MODIS and the other is VIIRS.
      2. Their WGS84 distance is within the spatial threshold.
      3. Their absolute acquisition-time difference is within the
         temporal window.

    Connected clusters represent graph associations only. They do not
    establish that observations are the same physical fire.
    """

    if distance_threshold_km <= 0:
        raise ValueError("distance_threshold_km must be greater than zero.")

    if time_window_minutes < 0:
        raise ValueError("time_window_minutes cannot be negative.")

    data = _validate_observations(observations)

    modis = data[data["sensor_family"] == "MODIS"]
    viirs = data[data["sensor_family"] == "VIIRS"]

    edge_records = []

    for modis_row, viirs_row in product(
        modis.to_dict("records"),
        viirs.to_dict("records"),
    ):
        distance = _distance_km(
            modis_row["longitude"],
            modis_row["latitude"],
            viirs_row["longitude"],
            viirs_row["latitude"],
        )

        signed_minutes = (
            viirs_row["acq_datetime_utc"]
            - modis_row["acq_datetime_utc"]
        ).total_seconds() / 60.0

        absolute_minutes = abs(signed_minutes)

        if (
            distance <= distance_threshold_km
            and absolute_minutes <= time_window_minutes
        ):
            edge_records.append({
                "modis_observation_id": modis_row["observation_id"],
                "viirs_observation_id": viirs_row["observation_id"],
                "modis_platform": modis_row["satellite_platform"],
                "viirs_platform": viirs_row["satellite_platform"],
                "distance_km": distance,
                "time_difference_minutes": absolute_minutes,
                "signed_time_difference_minutes": signed_minutes,
                "distance_threshold_km": distance_threshold_km,
                "time_window_minutes": time_window_minutes,
                "association_type": "candidate_only",
            })

    edge_columns = [
        "modis_observation_id",
        "viirs_observation_id",
        "modis_platform",
        "viirs_platform",
        "distance_km",
        "time_difference_minutes",
        "signed_time_difference_minutes",
        "distance_threshold_km",
        "time_window_minutes",
        "association_type",
    ]

    edges = pd.DataFrame(edge_records, columns=edge_columns)

    # Build connected components from candidate edges.
    participating_ids = set()
    union_find = _UnionFind(data["observation_id"].tolist())

    for edge in edge_records:
        modis_id = edge["modis_observation_id"]
        viirs_id = edge["viirs_observation_id"]

        participating_ids.update([modis_id, viirs_id])
        union_find.union(modis_id, viirs_id)

    components = {}
    for observation_id in participating_ids:
        root = union_find.find(observation_id)
        components.setdefault(root, []).append(observation_id)

    # Stable ordering makes cluster labels reproducible.
    component_members = sorted(
        [sorted(members) for members in components.values()],
        key=lambda members: members[0],
    )

    cluster_records = []

    for number, members in enumerate(component_members, start=1):
        member_set = set(members)

        modis_ids = sorted(
            data.loc[
                data["observation_id"].isin(member_set)
                & data["sensor_family"].eq("MODIS"),
                "observation_id",
            ].tolist()
        )

        viirs_ids = sorted(
            data.loc[
                data["observation_id"].isin(member_set)
                & data["sensor_family"].eq("VIIRS"),
                "observation_id",
            ].tolist()
        )

        internal_edges = [
            edge for edge in edge_records
            if (
                edge["modis_observation_id"] in member_set
                and edge["viirs_observation_id"] in member_set
            )
        ]

        modis_count = len(modis_ids)
        viirs_count = len(viirs_ids)

        if modis_count == 1 and viirs_count == 1:
            ambiguity = "one_to_one_candidate"
        elif modis_count == 1 and viirs_count > 1:
            ambiguity = "one_to_many"
        elif modis_count > 1 and viirs_count == 1:
            ambiguity = "many_to_one"
        else:
            ambiguity = "many_to_many"

        cluster_records.append({
            "cluster_id": f"FIRESYNC-{number:04d}",
            "modis_count": modis_count,
            "viirs_count": viirs_count,
            "observation_count": len(members),
            "candidate_edge_count": len(internal_edges),
            "ambiguity_type": ambiguity,
            "observation_ids": tuple(members),
            "modis_observation_ids": tuple(modis_ids),
            "viirs_observation_ids": tuple(viirs_ids),
            "association_status": "unconfirmed_candidate_cluster",
        })

    cluster_columns = [
        "cluster_id",
        "modis_count",
        "viirs_count",
        "observation_count",
        "candidate_edge_count",
        "ambiguity_type",
        "observation_ids",
        "modis_observation_ids",
        "viirs_observation_ids",
        "association_status",
    ]

    clusters = pd.DataFrame(cluster_records, columns=cluster_columns)

    unmatched = data[
        ~data["observation_id"].isin(participating_ids)
    ].copy()

    return AssociationResult(
        edges=edges,
        clusters=clusters,
        unmatched=unmatched,
    )
