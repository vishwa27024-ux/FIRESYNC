
import ast
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

INPUT = (
    ROOT / "data" / "processed"
    / "firms_archive_normalized_2021_2025.parquet"
)

BRIDGE_INPUT = (
    ROOT / "data" / "interim"
    / "association_bridge_edge_details.csv"
)

OUTPUT_DIR = ROOT / "data" / "interim"

DISTANCE_KM = 1.0
TIME_MINUTES = 60.0

# Exploratory reporting threshold, not a scientific cutoff.
HUB_DEGREE_THRESHOLD = 3


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


def main():
    from src.science.association_engine import build_associations

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    observations = pd.read_parquet(INPUT)
    observations["observation_id"] = (
        observations["observation_id"].astype(str)
    )

    result = build_associations(
        observations,
        distance_threshold_km=DISTANCE_KM,
        time_window_minutes=TIME_MINUTES,
    )

    observations = observations.set_index("observation_id")

    bridges = pd.read_csv(BRIDGE_INPUT)

    bridge_counts = Counter()
    for _, bridge in bridges.iterrows():
        bridge_counts[str(bridge["modis_observation_id"])] += 1
        bridge_counts[str(bridge["viirs_observation_id"])] += 1

    node_rows = []
    cluster_rows = []

    for _, cluster in result.clusters.iterrows():
        cluster_id = cluster["cluster_id"]
        node_ids = parse_ids(cluster["observation_ids"])
        node_set = set(str(x) for x in node_ids)

        edges = result.edges[
            result.edges["modis_observation_id"].astype(str).isin(node_set)
            & result.edges["viirs_observation_id"].astype(str).isin(node_set)
        ]

        neighbors = defaultdict(set)

        for _, edge in edges.iterrows():
            modis_id = str(edge["modis_observation_id"])
            viirs_id = str(edge["viirs_observation_id"])

            neighbors[modis_id].add(viirs_id)
            neighbors[viirs_id].add(modis_id)

        degrees = {
            node_id: len(neighbors[node_id])
            for node_id in node_set
        }

        hub_count = sum(
            degree >= HUB_DEGREE_THRESHOLD
            for degree in degrees.values()
        )

        max_degree = max(degrees.values(), default=0)

        cluster_rows.append({
            "cluster_id": cluster_id,
            "ambiguity_type": cluster["ambiguity_type"],
            "observation_count": len(node_set),
            "candidate_edge_count": len(edges),
            "maximum_observation_degree": max_degree,
            "mean_observation_degree": round(
                sum(degrees.values()) / len(degrees), 2
            ) if degrees else 0,
            "hub_observation_count": hub_count,
            "bridge_endpoint_count": sum(
                bridge_counts[node_id] > 0
                for node_id in node_set
            ),
        })

        for node_id, degree in degrees.items():
            if node_id not in observations.index:
                raise ValueError(
                    f"Observation not found: {node_id}"
                )

            obs = observations.loc[node_id]

            if isinstance(obs, pd.DataFrame):
                raise ValueError(
                    f"Duplicate observation ID: {node_id}"
                )

            opposite_neighbors = sorted(neighbors[node_id])

            row = {
                "cluster_id": cluster_id,
                "observation_id": node_id,
                "sensor_family": obs["sensor_family"],
                "satellite_platform": obs["satellite_platform"],
                "latitude": obs["latitude"],
                "longitude": obs["longitude"],
                "acq_datetime_utc": obs["acq_datetime_utc"],
                "degree": degree,
                "hub_flag": degree >= HUB_DEGREE_THRESHOLD,
                "bridge_incident_count": bridge_counts[node_id],
                "opposite_observation_ids": ";".join(
                    opposite_neighbors
                ),
            }

            for column in [
                "confidence_native",
                "confidence_scheme",
                "confidence_class",
                "confidence_numeric",
                "frp_mw",
                "scan_native",
                "track_native",
            ]:
                if column in observations.columns:
                    row[column] = obs[column]

            node_rows.append(row)

    node_report = pd.DataFrame(node_rows)
    cluster_report = pd.DataFrame(cluster_rows)

    node_report = node_report.sort_values(
        ["degree", "bridge_incident_count"],
        ascending=False,
    )

    cluster_report = cluster_report.sort_values(
        ["maximum_observation_degree", "candidate_edge_count"],
        ascending=False,
    )

    node_path = (
        OUTPUT_DIR / "observation_connectivity_details.csv"
    )
    cluster_path = (
        OUTPUT_DIR / "cluster_connectivity_summary.csv"
    )
    report_path = (
        OUTPUT_DIR / "observation_connectivity_report.json"
    )

    node_report.to_csv(node_path, index=False)
    cluster_report.to_csv(cluster_path, index=False)

    report = {
        "input_observations": len(observations),
        "candidate_edges": len(result.edges),
        "association_clusters": len(result.clusters),
        "observations_in_clusters": len(node_report),
        "unmatched_observations": len(result.unmatched),
        "maximum_degree": int(node_report["degree"].max()),
        "observations_with_degree_3_or_more": int(
            (node_report["degree"] >= HUB_DEGREE_THRESHOLD).sum()
        ),
        "observations_incident_to_bridge_edges": int(
            (node_report["bridge_incident_count"] > 0).sum()
        ),
        "hub_degree_threshold": HUB_DEGREE_THRESHOLD,
        "interpretation": (
            "Degree counts distinct opposite-sensor observations "
            "connected by candidate edges. A high degree indicates "
            "graph connectivity, not a confirmed fire or a better "
            "observation. Bridge incidence is a graph diagnostic."
        ),
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 68)
    print("FIRESYNC | OBSERVATION CONNECTIVITY AUDIT")
    print("=" * 68)

    print(json.dumps(report, indent=2))

    print("\nTOP 20 MOST CONNECTED OBSERVATIONS")
    columns = [
        "cluster_id",
        "observation_id",
        "sensor_family",
        "satellite_platform",
        "degree",
        "bridge_incident_count",
        "hub_flag",
    ]

    print(node_report[columns].head(20).to_string(index=False))

    print("\nCLUSTER CONNECTIVITY SUMMARY")
    print(cluster_report.to_string(index=False))

    print("\nOUTPUT FILES")
    print(node_path)
    print(cluster_path)
    print(report_path)


if __name__ == "__main__":
    main()
