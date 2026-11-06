
"""Spatially optimized MODIS-VIIRS candidate association engine."""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .association_engine import (
    AssociationResult,
    _UnionFind,
    _distance_km,
    _validate_observations,
)


EDGE_COLUMNS = [
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

CLUSTER_COLUMNS = [
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

# Deliberately smaller than the minimum WGS84 curvature radius.
# This makes the spherical search radius conservative: it can return
# extra candidates, but exact WGS84 verification decides every edge.
CONSERVATIVE_EARTH_RADIUS_M = 6_300_000.0


def _unit_sphere_coordinates(data):
    """Convert longitude/latitude to 3D unit-sphere coordinates."""

    lat = np.radians(data["latitude"].to_numpy(dtype=float))
    lon = np.radians(data["longitude"].to_numpy(dtype=float))

    cos_lat = np.cos(lat)

    return np.column_stack((
        cos_lat * np.cos(lon),
        cos_lat * np.sin(lon),
        np.sin(lat),
    ))


def _build_clusters(data, edge_records):
    """Build connected components from candidate edges."""

    participating_ids = set()
    union_find = _UnionFind(data["observation_id"].tolist())

    for edge in edge_records:
        modis_id = edge["modis_observation_id"]
        viirs_id = edge["viirs_observation_id"]

        participating_ids.update((modis_id, viirs_id))
        union_find.union(modis_id, viirs_id)

    components = {}

    for observation_id in participating_ids:
        root = union_find.find(observation_id)
        components.setdefault(root, []).append(observation_id)

    members_list = sorted(
        (sorted(members) for members in components.values()),
        key=lambda members: members[0],
    )

    records = []

    for number, members in enumerate(members_list, start=1):
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

        records.append({
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

    unmatched = data[
        ~data["observation_id"].isin(participating_ids)
    ].copy()

    clusters = pd.DataFrame(records, columns=CLUSTER_COLUMNS)

    return clusters, unmatched


def build_associations_optimized(
    observations: pd.DataFrame,
    distance_threshold_km: float = 1.0,
    time_window_minutes: float = 60.0,
) -> AssociationResult:
    """Build candidate associations using a spatial index.

    cKDTree generates a conservative set of spatial candidates.
    Every retained edge is subsequently checked using exact WGS84
    ellipsoidal distance and the requested UTC time window.

    Candidate associations do not confirm the same physical fire.
    """

    if distance_threshold_km <= 0:
        raise ValueError("distance_threshold_km must be greater than zero.")

    if time_window_minutes < 0:
        raise ValueError("time_window_minutes cannot be negative.")

    data = _validate_observations(observations)

    modis = data[data["sensor_family"] == "MODIS"].reset_index(drop=True)
    viirs = data[data["sensor_family"] == "VIIRS"].reset_index(drop=True)

    if modis.empty or viirs.empty:
        clusters, unmatched = _build_clusters(data, [])
        return AssociationResult(
            edges=pd.DataFrame(columns=EDGE_COLUMNS),
            clusters=clusters,
            unmatched=unmatched,
        )

    # Build a spatial index over VIIRS observations.
    viirs_xyz = _unit_sphere_coordinates(viirs)
    tree = cKDTree(viirs_xyz)

    # Convert the maximum requested distance into a conservative
    # chord radius on the unit sphere.
    angular_radius = (
        distance_threshold_km * 1000.0
        / CONSERVATIVE_EARTH_RADIUS_M
    )
    chord_radius = 2.0 * np.sin(angular_radius / 2.0)

    modis_xyz = _unit_sphere_coordinates(modis)

    # Spatial candidate lookup. Exact distance is checked below.
    nearby_indices = tree.query_ball_point(
        modis_xyz,
        r=chord_radius,
    )

    edge_records = []

    modis_rows = modis.to_dict("records")
    viirs_rows = viirs.to_dict("records")

    for modis_index, candidate_indices in enumerate(nearby_indices):
        modis_row = modis_rows[modis_index]

        # Stable ordering makes outputs reproducible.
        for viirs_index in sorted(candidate_indices):
            viirs_row = viirs_rows[viirs_index]

            signed_minutes = (
                viirs_row["acq_datetime_utc"]
                - modis_row["acq_datetime_utc"]
            ).total_seconds() / 60.0

            absolute_minutes = abs(signed_minutes)

            if absolute_minutes > time_window_minutes:
                continue

            distance = _distance_km(
                modis_row["longitude"],
                modis_row["latitude"],
                viirs_row["longitude"],
                viirs_row["latitude"],
            )

            # Exact ellipsoidal verification.
            if distance > distance_threshold_km:
                continue

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

    edges = pd.DataFrame(edge_records, columns=EDGE_COLUMNS)

    clusters, unmatched = _build_clusters(data, edge_records)

    return AssociationResult(
        edges=edges,
        clusters=clusters,
        unmatched=unmatched,
    )
