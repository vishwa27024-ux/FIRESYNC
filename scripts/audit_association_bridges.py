
import ast
import json
import sys
from pathlib import Path
from collections import defaultdict, deque

import pandas as pd
from pyproj import Geod


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.science.association_engine import build_associations


INPUT = ROOT / "data" / "processed" / "firms_archive_normalized_2021_2025.parquet"
OUTPUT_DIR = ROOT / "data" / "interim"

DISTANCE_THRESHOLD_KM = 1.0
TIME_WINDOW_MINUTES = 60.0

GEOD = Geod(ellps="WGS84")


def parse_ids(value):
    if isinstance(value, (list, tuple, set)):
        return list(value)

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple, set)):
                return list(parsed)
        except (ValueError, SyntaxError):
            pass

    return []


def calculate_cluster_extent(node_ids, observations):
    points = observations.loc[node_ids]

    max_distance = 0.0

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            a = points.iloc[i]
            b = points.iloc[j]

            _, _, distance_m = GEOD.inv(
                a["longitude"],
                a["latitude"],
                b["longitude"],
                b["latitude"],
            )

            max_distance = max(max_distance, distance_m / 1000)

    timestamps = pd.to_datetime(
        points["acq_datetime_utc"], utc=True
    )

    time_span = (
        timestamps.max() - timestamps.min()
    ).total_seconds() / 60

    return max_distance, time_span


def find_bridges(edge_frame):
    adjacency = defaultdict(list)

    for edge_id, row in edge_frame.iterrows():
        a = row["modis_observation_id"]
        b = row["viirs_observation_id"]

        adjacency[a].append((b, edge_id))
        adjacency[b].append((a, edge_id))

    discovery = {}
    low = {}
    bridges = set()
    timer = 0

    def dfs(node, parent_edge=None):
        nonlocal timer

        timer += 1
        discovery[node] = timer
        low[node] = timer

        for neighbor, edge_id in adjacency[node]:
            if edge_id == parent_edge:
                continue

            if neighbor not in discovery:
                dfs(neighbor, edge_id)

                low[node] = min(low[node], low[neighbor])

                if low[neighbor] > discovery[node]:
                    bridges.add(edge_id)

            else:
                low[node] = min(low[node], discovery[neighbor])

    for node in adjacency:
        if node not in discovery:
            dfs(node)

    return bridges, adjacency


def components_after_removal(adjacency, removed_edge):
    visited = set()
    component_sizes = []

    for start in adjacency:
        if start in visited:
            continue

        queue = deque([start])
        visited.add(start)
        size = 0

        while queue:
            node = queue.popleft()
            size += 1

            for neighbor, edge_id in adjacency[node]:
                if edge_id == removed_edge:
                    continue

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        component_sizes.append(size)

    return sorted(component_sizes, reverse=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    observations = pd.read_parquet(INPUT)

    result = build_associations(
        observations,
        distance_threshold_km=DISTANCE_THRESHOLD_KM,
        time_window_minutes=TIME_WINDOW_MINUTES,
    )

    observations = observations.set_index("observation_id")

    summary_rows = []
    bridge_rows = []

    for _, cluster in result.clusters.iterrows():
        cluster_id = cluster["cluster_id"]
        node_ids = parse_ids(cluster["observation_ids"])

        max_distance, time_span = calculate_cluster_extent(
            node_ids, observations
        )

        spatially_extended = (
            max_distance > DISTANCE_THRESHOLD_KM
        )
        temporally_extended = (
            time_span > TIME_WINDOW_MINUTES
        )

        if not (spatially_extended or temporally_extended):
            continue

        node_set = set(node_ids)

        cluster_edges = result.edges[
            result.edges["modis_observation_id"].isin(node_set)
            & result.edges["viirs_observation_id"].isin(node_set)
        ].copy().reset_index(drop=True)

        bridges, adjacency = find_bridges(cluster_edges)

        summary_rows.append({
            "cluster_id": cluster_id,
            "observation_count": len(node_ids),
            "edge_count": len(cluster_edges),
            "max_internal_distance_km": round(max_distance, 3),
            "time_span_minutes": round(time_span, 1),
            "spatially_extended": spatially_extended,
            "temporally_extended": temporally_extended,
            "bridge_edge_count": len(bridges),
        })

        for edge_id in sorted(bridges):
            edge = cluster_edges.loc[edge_id]

            component_sizes = components_after_removal(
                adjacency, edge_id
            )

            bridge_rows.append({
    "cluster_id": cluster_id,
    "modis_observation_id": edge["modis_observation_id"],
    "viirs_observation_id": edge["viirs_observation_id"],
    "modis_platform": edge["modis_platform"],
    "viirs_platform": edge["viirs_platform"],
    "distance_km": round(
        edge["distance_km"], 3
    ),
    "time_difference_minutes": round(
        edge["time_difference_minutes"], 1
    ),
    "components_after_removal": len(component_sizes),
    "component_sizes_after_removal": str(
        component_sizes
    ),
})

    summary = pd.DataFrame(summary_rows)
    bridge_details = pd.DataFrame(bridge_rows)

    summary_path = (
        OUTPUT_DIR / "association_extended_cluster_bridge_summary.csv"
    )
    bridge_path = (
        OUTPUT_DIR / "association_bridge_edge_details.csv"
    )

    summary.to_csv(summary_path, index=False)
    bridge_details.to_csv(bridge_path, index=False)

    report = {
        "input_observations": len(observations),
        "total_candidate_edges": len(result.edges),
        "total_clusters": len(result.clusters),
        "extended_clusters_audited": len(summary),
        "bridge_edges_found": len(bridge_details),
        "clusters_with_bridges": (
            int((summary["bridge_edge_count"] > 0).sum())
            if not summary.empty else 0
        ),
        "distance_threshold_km": DISTANCE_THRESHOLD_KM,
        "time_window_minutes": TIME_WINDOW_MINUTES,
        "interpretation": (
            "Bridge edges are graph connections whose removal "
            "splits a connected component. They remain valid "
            "pairwise candidates and are not confirmed fire matches."
        ),
    }

    report_path = (
        OUTPUT_DIR / "association_bridge_audit_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("FIRESYNC | ASSOCIATION BRIDGE-EDGE AUDIT")
    print("=" * 65)

    print(json.dumps(report, indent=2))

    print("\nEXTENDED CLUSTER SUMMARY")
    print(summary.to_string(index=False))

    print("\nBRIDGE EDGE DETAILS")
    if bridge_details.empty:
        print("No bridge edges found in extended clusters.")
    else:
        print(bridge_details.to_string(index=False))

    print("\nOUTPUT FILES")
    print(summary_path)
    print(bridge_path)
    print(report_path)


if __name__ == "__main__":
    main()
