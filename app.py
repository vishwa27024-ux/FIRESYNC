
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.science.association_engine import build_associations
from src.science.optimized_association_engine import (
    build_associations_optimized,
)


# ============================================================
# FIRESYNC
# Sensor-Aware Satellite Fire Intelligence System
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "firms_archive_normalized_2021_2025.parquet"
)

ENGINE_REPORT = (
    ROOT
    / "data"
    / "interim"
    / "historical_2021_2025_engine_comparison_report.json"
)

ENGINE_CSV = (
    ROOT
    / "data"
    / "interim"
    / "historical_2021_2025_engine_comparison.csv"
)

BOUNDARY_DOCS = [
    ROOT / "docs" / "boundary_provenance.md",
    ROOT / "docs" / "boundary_comparison.md",
]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FIRESYNC | Satellite Fire Intelligence",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# VISUAL DESIGN
# ============================================================

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500'
        '&family=Manrope:wght@400;500;600;700;800&display=swap'
    );

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }

    .stApp {
        background-color: #07111f;
        color: #e8f1fb;
    }

    [data-testid="stSidebar"] {
        background-color: #0c1929;
        border-right: 1px solid #1d344a;
    }

    [data-testid="stMetric"] {
        background-color: #102237;
        border: 1px solid #1e3a50;
        padding: 14px 16px;
        border-radius: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: #9eb5c9;
    }

    [data-testid="stMetricValue"] {
        color: #e8f7ff;
    }

    h1, h2, h3 {
        letter-spacing: -0.03em;
    }

    .eyebrow {
        color: #48d6c5;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.72rem;
        font-weight: 800;
    }

    .hero {
        padding: 1.4rem;
        border: 1px solid #1c4255;
        border-radius: 16px;
        background:
            linear-gradient(
                115deg,
                #10283a,
                #0c1c2d 65%,
                #102b35
            );
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin: 0.25rem 0;
    }

    .muted {
        color: #9eb5c9;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #1e3a50;
        border-radius: 10px;
        overflow: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner="Loading normalized NASA FIRMS data...")
def load_observations(path_string: str) -> pd.DataFrame:

    data = pd.read_parquet(path_string)

    data["acq_datetime_utc"] = pd.to_datetime(
        data["acq_datetime_utc"],
        utc=True,
        errors="coerce",
    )

    if "acq_date" in data.columns:

        data["acq_date"] = pd.to_datetime(
            data["acq_date"],
            errors="coerce",
        ).dt.date

    else:

        data["acq_date"] = (
            data["acq_datetime_utc"].dt.date
        )

    numeric_columns = [
        "latitude",
        "longitude",
        "frp_mw",
        "confidence_numeric",
    ]

    for column in numeric_columns:

        if column in data.columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

    return data


def safe_read_json(path: Path):

    if not path.exists():
        return None

    try:

        return json.loads(
            path.read_text(encoding="utf-8")
        )

    except Exception as exc:

        return {
            "read_error": str(exc)
        }


def observation_columns(data: pd.DataFrame):

    preferred = [
        "observation_id",
        "source_product",
        "satellite_platform",
        "sensor_family",
        "acq_datetime_utc",
        "latitude",
        "longitude",
        "confidence_native",
        "confidence_scheme",
        "frp_mw",
        "modis_brightness_21_22_k",
        "modis_brightness_t31_k",
        "viirs_brightness_i4_k",
        "viirs_brightness_i5_k",
        "scan_native",
        "track_native",
        "daynight",
    ]

    return [
        column
        for column in preferred
        if column in data.columns
    ]


def show_table(data: pd.DataFrame, height=350):

    if data is None or data.empty:

        st.info("No records are available for this selection.")

        return

    st.dataframe(
        data,
        use_container_width=True,
        height=height,
        hide_index=True,
    )


# ============================================================
# MAP GENERATOR
# ============================================================

def create_map(data: pd.DataFrame, height=620):

    if data.empty:
        return None

    hover_columns = [
        column
        for column in [
            "satellite_platform",
            "acq_date",
            "frp_mw",
            "confidence_native",
            "daynight",
            "source_product",
        ]
        if column in data.columns
    ]

    common_arguments = {
        "data_frame": data,
        "lat": "latitude",
        "lon": "longitude",
        "color": "sensor_family",
        "hover_data": hover_columns,
        "zoom": 8,
        "center": {
            "lat": 11.75,
            "lon": 76.55,
        },
        "height": height,
        "template": "plotly_dark",
    }

    if hasattr(px, "scatter_map"):

        fig = px.scatter_map(
            **common_arguments,
            map_style="open-street-map",
        )

    else:

        fig = px.scatter_mapbox(
            **common_arguments,
            mapbox_style="open-street-map",
        )

    fig.update_layout(
        margin={
            "r": 0,
            "t": 25,
            "l": 0,
            "b": 0,
        },
        paper_bgcolor="#07111f",
        plot_bgcolor="#07111f",
        legend_title_text="Sensor family",
    )

    return fig


# ============================================================
# ASSOCIATION ENGINE CONNECTION
# ============================================================

def run_association(
    data: pd.DataFrame,
    distance_km: float,
    time_minutes: float,
    engine: str,
):

    if engine == "Reference":

        result = build_associations(
            data,
            distance_threshold_km=distance_km,
            time_window_minutes=time_minutes,
        )

    else:

        result = build_associations_optimized(
            data,
            distance_threshold_km=distance_km,
            time_window_minutes=time_minutes,
        )

    return (
        result.edges,
        result.clusters,
        result.unmatched,
    )


# ============================================================
# START APPLICATION
# ============================================================

if not DATA_FILE.exists():

    st.error(
        "The normalized historical dataset was not found."
    )

    st.code(str(DATA_FILE))

    st.stop()


try:

    df_all = load_observations(
        str(DATA_FILE)
    )

except Exception as exc:

    st.error(
        f"Unable to load historical data: {exc}"
    )

    st.stop()


REQUIRED_COLUMNS = {
    "observation_id",
    "sensor_family",
    "satellite_platform",
    "latitude",
    "longitude",
    "acq_datetime_utc",
}

missing_columns = (
    REQUIRED_COLUMNS - set(df_all.columns)
)

if missing_columns:

    st.error(
        f"Dataset is missing columns: {sorted(missing_columns)}"
    )

    st.stop()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown(
    '<div class="eyebrow">NASA Space Apps research prototype</div>',
    unsafe_allow_html=True,
)

st.sidebar.title("🔥 FIRESYNC")

st.sidebar.caption(
    "Two satellites. Different views. One clearer picture."
)

st.sidebar.divider()

PAGES = [
    "Overview",
    "Fire Map Explorer",
    "Activity Calendar",
    "Association Explorer",
    "Harmonization Lab",
    "Activity Investigation",
    "Evidence Centre",
]

page = st.sidebar.radio(
    "NAVIGATION",
    PAGES,
)


# ============================================================
# GLOBAL FILTERS
# ============================================================

st.sidebar.divider()

st.sidebar.markdown("**GLOBAL FILTERS**")

platforms = sorted(
    df_all["satellite_platform"]
    .dropna()
    .astype(str)
    .unique()
)

sensors = sorted(
    df_all["sensor_family"]
    .dropna()
    .astype(str)
    .unique()
)

selected_platforms = st.sidebar.multiselect(
    "Satellite platforms",
    platforms,
    default=platforms,
)

selected_sensors = st.sidebar.multiselect(
    "Sensor families",
    sensors,
    default=sensors,
)

# ============================================================
# GLOBAL DATE FILTER — FULL CALENDAR YEAR SUPPORT
# ============================================================

min_date = (
    df_all["acq_datetime_utc"].min().date()
)

max_date = (
    df_all["acq_datetime_utc"].max().date()
)

# Allow selection of the complete calendar years represented
# in the historical dataset, not just dates with observations.

date_min_boundary = min_date.replace(
    month=1,
    day=1,
)

date_max_boundary = max_date.replace(
    month=12,
    day=31,
)

date_range = st.sidebar.date_input(
    "Acquisition date (UTC)",
    value=(min_date, max_date),
    min_value=date_min_boundary,
    max_value=date_max_boundary,
)

if isinstance(date_range, (tuple, list)):

    if len(date_range) == 2:

        start_date, end_date = date_range

    elif len(date_range) == 1:

        start_date = end_date = date_range[0]

    else:

        start_date, end_date = min_date, max_date

else:

    start_date = end_date = date_range


if start_date > end_date:

    start_date, end_date = end_date, start_date


mask = (
    df_all["satellite_platform"]
    .astype(str)
    .isin(selected_platforms)
    &
    df_all["sensor_family"]
    .astype(str)
    .isin(selected_sensors)
    &
    df_all["acq_datetime_utc"]
    .dt.date.between(start_date, end_date)
)

df = df_all.loc[mask].copy()


st.sidebar.caption(
    f"Filtered observations: {len(df):,} / {len(df_all):,}"
)

st.sidebar.caption(
    "All acquisition timestamps are interpreted as UTC."
)

st.sidebar.divider()

st.sidebar.caption(
    "Satellite thermal anomalies are not ground-confirmed fires."
)


# ============================================================
# APPLICATION HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="eyebrow">
            Sensor-aware satellite fire intelligence
        </div>

        <h1>FIRESYNC</h1>

        <div class="muted">
            Harmonizing MODIS and VIIRS observations through
            transparent evidence, candidate associations and uncertainty.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


if df.empty:

    st.warning(
        "No observations match the current filters. "
        "Change the sidebar selections."
    )

    st.stop()


# ============================================================
# MODULE 1 — OVERVIEW
# ============================================================

if page == "Overview":

    st.subheader(
        "Historical observation overview"
    )

    st.caption(
        "Normalized NASA FIRMS historical archive observations. "
        "Raw detection totals are descriptive and are not directly "
        "comparable measurements of fire activity."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total observations",
        f"{len(df):,}",
    )

    col2.metric(
        "MODIS observations",
        f"{(df['sensor_family'] == 'MODIS').sum():,}",
    )

    col3.metric(
        "VIIRS observations",
        f"{(df['sensor_family'] == 'VIIRS').sum():,}",
    )

    col4.metric(
        "Satellite platforms",
        df["satellite_platform"].nunique(),
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.markdown(
            "#### Observations by platform"
        )

        platform_counts = (
            df.groupby(
                ["satellite_platform", "sensor_family"]
            )
            .size()
            .reset_index(name="Observations")
        )

        fig = px.bar(
            platform_counts,
            x="satellite_platform",
            y="Observations",
            color="sensor_family",
            barmode="group",
            template="plotly_dark",
        )

        fig.update_layout(
            paper_bgcolor="#07111f",
            plot_bgcolor="#07111f",
            xaxis_title="Satellite platform",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:

        st.markdown(
            "#### Observations by year"
        )

        yearly = (
            df.assign(
                year=df["acq_datetime_utc"].dt.year
            )
            .groupby(
                ["year", "sensor_family"]
            )
            .size()
            .reset_index(name="Observations")
        )

        fig = px.bar(
            yearly,
            x="year",
            y="Observations",
            color="sensor_family",
            barmode="group",
            template="plotly_dark",
        )

        fig.update_layout(
            paper_bgcolor="#07111f",
            plot_bgcolor="#07111f",
            xaxis_title="UTC year",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.markdown(
        "#### Monthly retrieval pattern"
    )

    monthly = (
        df.assign(
            month=df["acq_datetime_utc"]
            .dt.to_period("M")
            .astype(str)
        )
        .groupby(
            ["month", "sensor_family"]
        )
        .size()
        .reset_index(name="Observations")
    )

    fig = px.line(
        monthly,
        x="month",
        y="Observations",
        color="sensor_family",
        markers=True,
        template="plotly_dark",
    )

    fig.update_layout(
        paper_bgcolor="#07111f",
        plot_bgcolor="#07111f",
        xaxis_title="Month (UTC)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.markdown(
        "#### Platform summary"
    )

    summary = (
        df.groupby(
            ["satellite_platform", "sensor_family"],
            dropna=False,
        )
        .agg(
            observations=("observation_id", "count"),
            first_observation=("acq_datetime_utc", "min"),
            last_observation=("acq_datetime_utc", "max"),
            median_frp_mw=("frp_mw", "median"),
        )
        .reset_index()
    )

    show_table(summary, 300)

    st.info(
        "A date with zero retrieved records does not prove that "
        "no fire occurred or that satellite observation coverage "
        "was available. Counts also reflect sensor resolution, "
        "acquisition opportunities and archive characteristics."
    )


# ============================================================
# MODULE 2 — FIRE MAP EXPLORER
# ============================================================

elif page == "Fire Map Explorer":

    st.subheader(
        "Fire Map Explorer"
    )

    st.caption(
        "Explore the locations of archived satellite thermal-anomaly observations."
    )

    st.warning(
        "Each point represents an individual satellite observation. "
        "Nearby points may be repeated detections or different observations. "
        "They are not automatically unique fires."
    )

    map_df = df.copy()

    maximum_points = min(
        5000,
        max(100, len(map_df)),
    )

    default_points = min(
        2000,
        maximum_points,
    )

    max_points = st.slider(
        "Maximum points to display",
        min_value=100,
        max_value=maximum_points,
        value=default_points,
        step=100,
    )

    if len(map_df) > max_points:

        map_df = map_df.sample(
            max_points,
            random_state=42,
        )

    fig = create_map(map_df)

    if fig is not None:

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.caption(
        f"Displaying {len(map_df):,} of {len(df):,} filtered observations."
    )

    st.markdown(
        "#### Observation records"
    )

    show_table(
        df[
            observation_columns(df)
        ].sort_values(
            "acq_datetime_utc",
            ascending=False,
        ),
        450,
    )


# ============================================================
# MODULE 3 — ACTIVITY CALENDAR
# ============================================================

elif page == "Activity Calendar":

    st.subheader(
        "Historical activity calendar"
    )

    st.caption(
        "A calendar of retrieved detections separated by sensor family. "
        "This is not a ground-truth fire-occurrence calendar."
    )

    calendar_df = df.copy()

    calendar_df["date"] = (
        calendar_df["acq_datetime_utc"]
        .dt.floor("D")
    )

    daily = (
        calendar_df.groupby(
            ["date", "sensor_family"]
        )
        .size()
        .reset_index(name="observations")
    )

    left, right = st.columns(2)

    with left:

        st.markdown(
            "#### Daily observations"
        )

        fig = px.line(
            daily,
            x="date",
            y="observations",
            color="sensor_family",
            markers=True,
            template="plotly_dark",
        )

        fig.update_layout(
            paper_bgcolor="#07111f",
            plot_bgcolor="#07111f",
            xaxis_title="Acquisition date (UTC)",
            yaxis_title="Retrieved observations",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:

        st.markdown(
            "#### Monthly observations"
        )

        calendar_df["month"] = (
            calendar_df["acq_datetime_utc"]
            .dt.to_period("M")
            .astype(str)
        )

        monthly = (
            calendar_df.groupby(
                ["month", "sensor_family"]
            )
            .size()
            .reset_index(name="observations")
        )

        fig = px.bar(
            monthly,
            x="month",
            y="observations",
            color="sensor_family",
            barmode="group",
            template="plotly_dark",
        )

        fig.update_layout(
            paper_bgcolor="#07111f",
            plot_bgcolor="#07111f",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.markdown(
        "#### Daily retrieval table"
    )

    pivot = (
        daily.pivot(
            index="date",
            columns="sensor_family",
            values="observations",
        )
        .fillna(0)
        .astype(int)
    )

    pivot["Total"] = pivot.sum(axis=1)

    show_table(
        pivot.reset_index().sort_values(
            "date",
            ascending=False,
        ),
        420,
    )

    st.info(
        "Zero or missing dates indicate zero records in this "
        "filtered archive extract. They do not establish absence "
        "of fire or prove satellite non-coverage."
    )


# ============================================================
# MODULE 4 — ASSOCIATION EXPLORER
# ============================================================

elif page == "Association Explorer":

    st.subheader(
        "MODIS–VIIRS candidate association explorer"
    )

    st.caption(
        "Spatially and temporally constrained candidate links. "
        "No link is a confirmed same-fire determination."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        distance = st.slider(
            "Maximum separation (km)",
            0.1,
            10.0,
            1.0,
            0.1,
        )

    with col2:

        time_window = st.slider(
            "Maximum time difference (minutes)",
            0,
            180,
            60,
            5,
        )

    with col3:

        engine = st.selectbox(
            "Association engine",
            ["Optimized", "Reference"],
        )

    if st.button(
        "Run candidate association",
        type="primary",
    ):

        st.session_state["association_parameters"] = (
            distance,
            time_window,
            engine,
        )

        st.session_state.pop(
            "association_result",
            None,
        )

    parameters = st.session_state.get(
        "association_parameters",
        (distance, time_window, engine),
    )

    run_distance, run_time, run_engine = parameters

    should_run = (
        "association_result" not in st.session_state
        or st.session_state.get("association_filter_signature")
        != (
            len(df),
            str(start_date),
            str(end_date),
            tuple(selected_platforms),
            tuple(selected_sensors),
        )
    )

    if should_run:

        try:

            with st.spinner(
                f"Running {run_engine.lower()} association engine..."
            ):

                result = run_association(
                    df,
                    float(run_distance),
                    float(run_time),
                    run_engine,
                )

                st.session_state["association_result"] = result

                st.session_state["association_filter_signature"] = (
                    len(df),
                    str(start_date),
                    str(end_date),
                    tuple(selected_platforms),
                    tuple(selected_sensors),
                )

        except Exception as exc:

            st.error(
                f"Association engine failed: {exc}"
            )

            st.stop()

    edges, clusters, unmatched = (
        st.session_state["association_result"]
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Candidate edges",
        f"{len(edges):,}",
    )

    m2.metric(
        "Candidate clusters",
        f"{len(clusters):,}",
    )

    m3.metric(
        "Unmatched observations",
        f"{len(unmatched):,}",
    )

    m4.metric(
        "Observations in clusters",
        f"{len(df) - len(unmatched):,}",
    )

    st.caption(
        f"Thresholds: {run_distance:g} km / "
        f"{run_time:g} minutes | Engine: {run_engine}"
    )

    st.divider()

    tab1, tab2, tab3 = st.tabs(
        [
            "Candidate edges",
            "Clusters",
            "Unmatched",
        ]
    )

    with tab1:

        st.markdown(
            "#### Candidate cross-sensor edges"
        )

        edge_view = (
            edges.sort_values("distance_km")
            if not edges.empty
            else edges
        )

        show_table(edge_view, 420)

        if not edges.empty:

            st.download_button(
                "Download candidate edges CSV",
                edges.to_csv(index=False),
                "firesync_candidate_edges.csv",
                "text/csv",
            )

    with tab2:

        st.markdown(
            "#### Connected candidate clusters"
        )

        show_table(clusters, 380)

        if not clusters.empty:

            ambiguity = (
                clusters["ambiguity_type"]
                .value_counts()
                .rename_axis("Ambiguity")
                .reset_index(name="Clusters")
            )

            fig = px.bar(
                ambiguity,
                x="Ambiguity",
                y="Clusters",
                color="Ambiguity",
                template="plotly_dark",
            )

            fig.update_layout(
                paper_bgcolor="#07111f",
                plot_bgcolor="#07111f",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    with tab3:

        st.markdown(
            "#### Observations without candidate edges"
        )

        show_table(
            unmatched[
                observation_columns(unmatched)
            ],
            420,
        )

    st.warning(
        "Connected components are graph structures. Transitive "
        "links can connect observations whose overall cluster span "
        "exceeds the pairwise threshold. Candidate edges and clusters "
        "are not confirmed physical fires."
    )


# ============================================================
# MODULE 5 — HARMONIZATION LAB
# ============================================================

elif page == "Harmonization Lab":

    st.subheader(
        "Harmonization Lab"
    )

    st.caption(
        "Inspect sensor differences, candidate matching sensitivity "
        "and confidence semantics without forcing unlike measurements "
        "into one scale."
    )

    modis = df[
        df["sensor_family"] == "MODIS"
    ]

    viirs = df[
        df["sensor_family"] == "VIIRS"
    ]

    left, right = st.columns(2)

    left.metric(
        "MODIS records",
        f"{len(modis):,}",
    )

    right.metric(
        "VIIRS records",
        f"{len(viirs):,}",
    )

    st.divider()

    st.markdown(
        "#### Native confidence representation"
    )

    if {
        "confidence_scheme",
        "confidence_native",
    }.issubset(df.columns):

        confidence = (
            df.groupby(
                [
                    "sensor_family",
                    "confidence_scheme",
                    "confidence_native",
                ],
                dropna=False,
            )
            .size()
            .reset_index(name="observations")
        )

        show_table(confidence, 340)

    else:

        st.info(
            "Confidence scheme fields are unavailable."
        )

    st.markdown(
        "#### Sensor-specific measurement channels"
    )

    definitions = [
        (
            "MODIS",
            "modis_brightness_21_22_k",
            "MODIS brightness channel 21/22",
        ),
        (
            "MODIS",
            "modis_brightness_t31_k",
            "MODIS brightness channel T31",
        ),
        (
            "VIIRS",
            "viirs_brightness_i4_k",
            "VIIRS brightness channel I4",
        ),
        (
            "VIIRS",
            "viirs_brightness_i5_k",
            "VIIRS brightness channel I5",
        ),
    ]

    channel_records = []

    for sensor, column, description in definitions:

        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df.loc[
                df["sensor_family"] == sensor,
                column,
            ],
            errors="coerce",
        )

        channel_records.append(
            {
                "Sensor": sensor,
                "Native field": column,
                "Description": description,
                "Non-null values": int(values.notna().sum()),
                "Median": values.median(),
            }
        )

    show_table(
        pd.DataFrame(channel_records),
        280,
    )

    st.info(
        "MODIS confidence is numeric, whereas VIIRS confidence "
        "is categorical (l/n/h). The schemes are not interchangeable. "
        "Sensor-specific brightness channels are preserved separately; "
        "this application does not claim radiometric calibration."
    )

    st.divider()

    st.markdown(
        "#### Candidate-association sensitivity"
    )

    sensitivity_time = st.slider(
        "Time window (minutes)",
        0,
        180,
        60,
        5,
        key="sensitivity_time",
    )

    thresholds = st.multiselect(
        "Distance thresholds (km)",
        [0.5, 1.0, 2.0, 5.0],
        default=[0.5, 1.0, 2.0],
    )

    if st.button(
        "Calculate sensitivity",
        type="primary",
    ):

        if not thresholds:

            st.warning(
                "Select at least one distance threshold."
            )

        else:

            records = []

            progress = st.progress(0)

            for index, threshold in enumerate(thresholds):

                result = build_associations_optimized(
                    df,
                    float(threshold),
                    float(sensitivity_time),
                )

                records.append(
                    {
                        "Distance threshold (km)": threshold,
                        "Time window (minutes)": sensitivity_time,
                        "Candidate edges": len(result.edges),
                        "Connected clusters": len(result.clusters),
                        "Observations in clusters": (
                            len(df) - len(result.unmatched)
                        ),
                        "Unmatched observations": len(result.unmatched),
                    }
                )

                progress.progress(
                    (index + 1) / len(thresholds)
                )

            st.session_state["sensitivity_results"] = (
                pd.DataFrame(records)
            )

    if "sensitivity_results" in st.session_state:

        sensitivity = st.session_state[
            "sensitivity_results"
        ]

        show_table(
            sensitivity,
            260,
        )

        fig = px.line(
            sensitivity,
            x="Distance threshold (km)",
            y="Candidate edges",
            markers=True,
            template="plotly_dark",
        )

        fig.update_layout(
            paper_bgcolor="#07111f",
            plot_bgcolor="#07111f",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.divider()

    st.markdown(
        "#### Reference versus optimized engine benchmark"
    )

    st.caption(
        "This is an on-demand runtime comparison on the currently "
        "filtered observations. It also checks whether the two engines "
        "produce identical outputs for the selected thresholds."
    )

    benchmark_distance = st.number_input(
        "Benchmark distance (km)",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        key="benchmark_distance",
    )

    benchmark_time = st.number_input(
        "Benchmark time window (minutes)",
        min_value=0,
        max_value=180,
        value=60,
        step=5,
        key="benchmark_time",
    )

    if st.button(
        "Compare both engines",
        type="primary",
    ):

        try:

            start = time.perf_counter()

            reference = build_associations(
                df,
                benchmark_distance,
                benchmark_time,
            )

            reference_seconds = (
                time.perf_counter() - start
            )

            start = time.perf_counter()

            optimized = build_associations_optimized(
                df,
                benchmark_distance,
                benchmark_time,
            )

            optimized_seconds = (
                time.perf_counter() - start
            )

            reference_edges = reference.edges
            optimized_edges = optimized.edges

            sort_columns = [
                "modis_observation_id",
                "viirs_observation_id",
            ]

            reference_edges = (
                reference_edges.sort_values(sort_columns)
                .reset_index(drop=True)
            )

            optimized_edges = (
                optimized_edges.sort_values(sort_columns)
                .reset_index(drop=True)
            )

            edge_ids_equal = (
                reference_edges[sort_columns].equals(
                    optimized_edges[sort_columns]
                )
            )

            reference_clusters = (
                reference.clusters.sort_values("cluster_id")
                .reset_index(drop=True)
            )

            optimized_clusters = (
                optimized.clusters.sort_values("cluster_id")
                .reset_index(drop=True)
            )

            cluster_ids_equal = (
                reference_clusters["observation_ids"].tolist()
                == optimized_clusters["observation_ids"].tolist()
            )

            unmatched_equal = (
                set(reference.unmatched["observation_id"])
                == set(optimized.unmatched["observation_id"])
            )

            if optimized_seconds > 0:

                speedup = (
                    reference_seconds / optimized_seconds
                )

            else:

                speedup = float("nan")

            benchmark_result = {
                "Reference runtime (seconds)": reference_seconds,
                "Optimized runtime (seconds)": optimized_seconds,
                "Measured speedup": speedup,
                "Reference edges": len(reference.edges),
                "Optimized edges": len(optimized.edges),
                "Reference clusters": len(reference.clusters),
                "Optimized clusters": len(optimized.clusters),
                "Candidate edge IDs identical": edge_ids_equal,
                "Cluster memberships identical": cluster_ids_equal,
                "Unmatched IDs identical": unmatched_equal,
            }

            st.session_state["benchmark_result"] = (
                benchmark_result
            )

        except Exception as exc:

            st.error(
                f"Benchmark failed: {exc}"
            )

    if "benchmark_result" in st.session_state:

        benchmark_result = st.session_state[
            "benchmark_result"
        ]

        st.json(
            benchmark_result,
            expanded=True,
        )

        identical = all(
            benchmark_result[key]
            for key in [
                "Candidate edge IDs identical",
                "Cluster memberships identical",
                "Unmatched IDs identical",
            ]
        )

        if identical:

            st.success(
                "The engines returned identical candidate edge IDs, "
                "cluster memberships and unmatched IDs for this run."
            )

        else:

            st.error(
                "The engine outputs differ. Investigate before relying "
                "on the optimized result."
            )

        st.caption(
            "Runtime measurements vary with machine load, data size "
            "and execution order. They are not a global scalability claim."
        )

    st.warning(
        "Sensitivity and benchmark outputs describe algorithm behavior "
        "under selected conditions. They do not establish an optimal "
        "scientific threshold or confirm physical fire events."
    )


# ============================================================
# MODULE 6 — ACTIVITY INVESTIGATION
# ============================================================

elif page == "Activity Investigation":

    st.subheader(
        "Activity Investigation"
    )

    st.caption(
        "Inspect individual observations or explore their candidate "
        "association clusters."
    )

    st.markdown(
        "#### Search an observation"
    )

    observation_query = st.text_input(
        "Exact observation ID",
        placeholder="Paste an observation_id",
    )

    if observation_query.strip():

        matches = df[
            df["observation_id"].astype(str)
            == observation_query.strip()
        ]

        if matches.empty:

            st.warning(
                "No exact observation ID was found in the currently "
                "filtered dataset. Check the ID or global filters."
            )

        else:

            st.markdown(
                "#### Selected observation"
            )

            show_table(
                matches[
                    observation_columns(matches)
                ],
                240,
            )

            fig = create_map(
                matches,
                420,
            )

            if fig is not None:

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

    st.divider()

    st.markdown(
        "#### Candidate cluster investigation"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        investigation_distance = st.number_input(
            "Distance threshold (km)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            key="investigation_distance",
        )

    with col2:

        investigation_time = st.number_input(
            "Time window (minutes)",
            min_value=0,
            max_value=180,
            value=60,
            step=5,
            key="investigation_time",
        )

    with col3:

        investigation_engine = st.selectbox(
            "Engine",
            ["Optimized", "Reference"],
            key="investigation_engine",
        )

    if st.button(
        "Build investigation clusters",
        type="primary",
    ):

        try:

            result = run_association(
                df,
                float(investigation_distance),
                float(investigation_time),
                investigation_engine,
            )

            st.session_state["investigation_result"] = (
                result
            )

            st.session_state["investigation_filter_signature"] = (
                len(df),
                str(start_date),
                str(end_date),
                tuple(selected_platforms),
                tuple(selected_sensors),
            )

        except Exception as exc:

            st.error(
                f"Investigation failed: {exc}"
            )

    current_signature = (
        len(df),
        str(start_date),
        str(end_date),
        tuple(selected_platforms),
        tuple(selected_sensors),
    )

    if (
        "investigation_result" in st.session_state
        and st.session_state.get("investigation_filter_signature")
        == current_signature
    ):

        edges, clusters, unmatched = (
            st.session_state["investigation_result"]
        )

        if clusters.empty:

            st.info(
                "No candidate clusters were found with these "
                "filters and thresholds."
            )

        else:

            cluster_options = (
                clusters["cluster_id"].tolist()
            )

            chosen_cluster = st.selectbox(
                "Choose a candidate cluster",
                cluster_options,
            )

            cluster_row = clusters.loc[
                clusters["cluster_id"] == chosen_cluster
            ].iloc[0]

            member_ids = list(
                cluster_row["observation_ids"]
            )

            cluster_observations = df[
                df["observation_id"].isin(member_ids)
            ].copy()

            cluster_edges = edges[
                edges["modis_observation_id"].isin(member_ids)
                &
                edges["viirs_observation_id"].isin(member_ids)
            ].copy()

            a, b, c, d = st.columns(4)

            a.metric(
                "MODIS members",
                int(cluster_row["modis_count"]),
            )

            b.metric(
                "VIIRS members",
                int(cluster_row["viirs_count"]),
            )

            c.metric(
                "Candidate edges",
                int(cluster_row["candidate_edge_count"]),
            )

            d.metric(
                "Ambiguity",
                str(cluster_row["ambiguity_type"]),
            )

            fig = create_map(
                cluster_observations,
                480,
            )

            if fig is not None:

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            st.markdown(
                "#### Cluster observations"
            )

            show_table(
                cluster_observations[
                    observation_columns(cluster_observations)
                ],
                350,
            )

            st.markdown(
                "#### Internal candidate edges"
            )

            show_table(
                cluster_edges,
                300,
            )

            st.warning(
                "This is an unconfirmed connected candidate group. "
                "It may contain one-to-many, many-to-one, many-to-many "
                "or transitive relationships. It is not a confirmed "
                "single fire."
            )


# ============================================================
# MODULE 7 — EVIDENCE CENTRE
# ============================================================

elif page == "Evidence Centre":

    st.subheader(
        "Evidence Centre"
    )

    st.caption(
        "Dataset provenance, processing evidence, methodology "
        "and known limitations."
    )

    st.markdown(
        "#### Dataset identity"
    )

    provenance = {
        "Dataset": DATA_FILE.name,
        "Records": f"{len(df_all):,}",
        "Columns": len(df_all.columns),
        "First UTC timestamp": str(
            df_all["acq_datetime_utc"].min()
        ),
        "Last UTC timestamp": str(
            df_all["acq_datetime_utc"].max()
        ),
        "Source products": (
            ", ".join(
                sorted(
                    df_all["source_product"]
                    .dropna()
                    .astype(str)
                    .unique()
                )
            )
            if "source_product" in df_all.columns
            else "Not available"
        ),
    }

    show_table(
        pd.DataFrame(
            [
                {
                    "Field": key,
                    "Value": value,
                }
                for key, value in provenance.items()
            ]
        ),
        280,
    )

    st.markdown(
        "#### Platform coverage in normalized archive"
    )

    platform_summary = (
        df_all.groupby(
            ["sensor_family", "satellite_platform"],
            dropna=False,
        )
        .agg(
            observations=("observation_id", "count"),
            first_utc=("acq_datetime_utc", "min"),
            last_utc=("acq_datetime_utc", "max"),
        )
        .reset_index()
    )

    show_table(
        platform_summary,
        300,
    )

    st.divider()

    st.markdown(
        "#### Historical engine benchmark evidence"
    )

    report = safe_read_json(
        ENGINE_REPORT
    )

    if report is not None:

        st.json(
            report,
            expanded=False,
        )

    elif ENGINE_CSV.exists():

        try:

            benchmark = pd.read_csv(
                ENGINE_CSV
            )

            show_table(
                benchmark,
                350,
            )

        except Exception as exc:

            st.warning(
                f"Benchmark CSV could not be read: {exc}"
            )

    else:

        st.info(
            "Historical engine benchmark report was not found "
            "at the expected path. The application does not "
            "invent or recreate the historical benchmark."
        )

    st.divider()

    st.markdown(
        "#### Boundary provenance documentation"
    )

    for document in BOUNDARY_DOCS:

        if document.exists():

            with st.expander(document.name):

                st.markdown(
                    document.read_text(
                        encoding="utf-8"
                    )
                )

        else:

            st.caption(
                f"Not found: {document.relative_to(ROOT)}"
            )

    st.divider()

    st.markdown(
        "#### Scientific safeguards"
    )

    st.markdown(
        """
        **Sensor differences**

        - MODIS nominal active-fire detection resolution is approximately 1 km.
        - VIIRS nominal active-fire detection resolution is approximately 375 m.
        - Raw MODIS and VIIRS detection counts are not directly equivalent.
        - MODIS numeric confidence and VIIRS categorical confidence remain separate.
        - Sensor-specific brightness channels remain separate.

        **Association limitations**

        - Candidate links use spatial and temporal thresholds.
        - Candidate links do not confirm the same physical fire.
        - Connected components may exhibit transitive chaining.
        - Cluster-wide extents may exceed an individual edge threshold.
        - One-to-many and many-to-many relationships are retained.

        **Spatial limitations**

        - Native scan and track values are preserved.
        - Scan and track are not treated as measured pixel-footprint geometry.
        - H3 or grid aggregation is not equivalent to sensor pixel geometry.
        - The Bandipur landscape acquisition rectangle is not an official reserve boundary.

        **Interpretation limitations**

        - A zero-record date is not proof of no fire.
        - A zero-record date does not establish sensor non-coverage.
        - Burned-area products are contextual evidence, not ground truth for individual detections.
        - Satellite thermal anomalies are not ground-confirmed wildfire events.
        """
    )

    st.divider()

    st.markdown(
        "#### Full normalized dataset schema"
    )

    schema = pd.DataFrame(
        {
            "Column": df_all.columns,
            "Data type": df_all.dtypes.astype(str).values,
            "Non-null values": [
                int(df_all[column].notna().sum())
                for column in df_all.columns
            ],
        }
    )

    show_table(
        schema,
        460,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "FIRESYNC | NASA FIRMS historical archive | MODIS + VIIRS | "
    "Research prototype. Satellite thermal anomalies are not "
    "ground-confirmed fire events."
)
