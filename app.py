
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.science.association_engine import build_associations
from src.science.optimized_association_engine import (
    build_associations_optimized,
)
from src.science.harmonization_engine import (
    build_daily_analytical_units,
    evaluate_harmonization_models,
    generate_harmonized_dataset,
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
    ROOT / "docs" / "harmonization_methodology.md",
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
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --fs-bg: #060d17;
        --fs-panel: #0d1b2a;
        --fs-panel-2: #112438;
        --fs-border: #20384d;
        --fs-text: #e8f1fb;
        --fs-muted: #91a8bc;
        --fs-cyan: #22d3ee;
        --fs-amber: #f5a742;
        --fs-teal: #48d6c5;
    }

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(23, 67, 89, .20), transparent 38%),
            linear-gradient(180deg, #07111f 0%, #050b14 100%);
        color: var(--fs-text);
    }

    [data-testid="stHeader"] {
        background: rgba(5, 11, 20, .72);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1928 0%, #08131f 100%);
        border-right: 1px solid var(--fs-border);
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding-top: 5px;
        padding-bottom: 5px;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(17, 36, 56, .96), rgba(10, 25, 40, .96));
        border: 1px solid #234257;
        padding: 15px 17px;
        border-radius: 13px;
        min-height: 102px;
    }

    [data-testid="stMetricLabel"] {
        color: #9eb5c9;
        font-size: .78rem;
        font-weight: 700;
    }

    [data-testid="stMetricValue"] {
        color: #effaff;
        font-weight: 800;
        letter-spacing: -.03em;
    }

    h1, h2, h3, h4 {
        letter-spacing: -.035em;
    }

    h1 { font-weight: 800; }
    h2, h3 { font-weight: 750; }

    .eyebrow {
        color: var(--fs-teal);
        text-transform: uppercase;
        letter-spacing: .17em;
        font-size: .68rem;
        font-weight: 800;
    }

    .hero {
        padding: 1.45rem 1.55rem;
        border: 1px solid #1c4255;
        border-radius: 17px;
        background:
            radial-gradient(ellipse at 90% 0%, rgba(43, 187, 190, .14), transparent 45%),
            linear-gradient(115deg, #10283a, #0c1c2d 65%, #102b35);
        margin-bottom: 1.25rem;
        box-shadow: 0 18px 45px rgba(0, 0, 0, .14);
    }

    .hero h1 { margin: .25rem 0 .35rem 0; }

    .muted {
        color: var(--fs-muted);
        line-height: 1.65;
    }

    .section-intro {
        padding: 1.15rem 1.35rem 1.2rem 1.35rem;
        margin: .15rem 0 1.1rem 0;
        border: 1px solid #1d3b51;
        border-radius: 15px;
        background:
            radial-gradient(ellipse at 100% 0%, rgba(34, 211, 238, .09), transparent 48%),
            linear-gradient(110deg, rgba(14, 33, 50, .96), rgba(9, 23, 37, .96));
    }

    .section-intro .section-eyebrow {
        color: #54d6d0;
        letter-spacing: .16em;
        font-size: .66rem;
        font-weight: 800;
        text-transform: uppercase;
    }

    .section-intro h2 {
        margin: .25rem 0 .35rem 0;
        font-size: 1.75rem;
        color: #eff8ff;
        font-weight: 800;
    }

    .section-intro p {
        color: #9cb2c4;
        margin: 0;
        line-height: 1.6;
        font-size: .92rem;
    }

    .section-kicker {
        color: #67cad7;
        font-family: 'DM Mono', monospace;
        font-size: .68rem;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin-bottom: .35rem;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #1e3a50;
        border-radius: 11px;
        overflow: hidden;
        background: rgba(8, 20, 33, .85);
    }

    div[data-testid="stPlotlyChart"] {
        border: 1px solid rgba(36, 67, 88, .72);
        border-radius: 13px;
        overflow: hidden;
        background: rgba(7, 17, 29, .65);
    }

    [data-testid="stExpander"] {
        border: 1px solid #20384d;
        border-radius: 12px;
        background: rgba(10, 25, 40, .58);
    }

    [data-testid="stTabs"] [role="tab"] {
        font-weight: 700;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 9px;
        border: 1px solid #2c6073;
        font-weight: 750;
        transition: all .18s ease;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #087f91, #0c9b9a);
        border: 1px solid #36c5c1;
        color: #f4ffff;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: #55d6d2;
        color: #ffffff;
        transform: translateY(-1px);
    }

    .stCaption, [data-testid="stCaptionContainer"] {
        color: #879fb3;
    }

    hr {
        border-color: #1c354a;
    }

    code, pre {
        font-family: 'DM Mono', monospace;
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


def render_module_header(eyebrow: str, title: str, subtitle: str):
    """Render a consistent, presentation-ready header for each module."""

    st.markdown(
        f"""
        <section class="section-intro">
            <div class="section-eyebrow">{eyebrow}</div>
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def style_figure(fig, height=None):
    """Apply FIRESYNC's shared dark scientific chart styling."""

    layout = {
        "template": "plotly_dark",
        "paper_bgcolor": "#07111f",
        "plot_bgcolor": "#091827",
        "font": {
            "family": "Manrope, sans-serif",
            "color": "#dceaf5",
            "size": 12,
        },
        "margin": {"l": 18, "r": 18, "t": 38, "b": 18},
        "legend": {
            "bgcolor": "rgba(7, 17, 31, 0.72)",
            "bordercolor": "#244258",
            "borderwidth": 1,
            "font": {"color": "#c5d6e3", "size": 11},
        },
        "hoverlabel": {
            "bgcolor": "#10283a",
            "bordercolor": "#3b7184",
            "font": {"color": "#effaff", "family": "Manrope"},
        },
    }
    if height is not None:
        layout["height"] = height

    fig.update_layout(**layout)

    trace_types = {
        getattr(trace, "type", None)
        for trace in fig.data
    }
    is_map_figure = any(
        trace_type in {"scattermap", "scattermapbox"}
        for trace_type in trace_types
    )

    if not is_map_figure:
        fig.update_xaxes(
            gridcolor="rgba(120, 155, 180, 0.13)",
            zerolinecolor="rgba(120, 155, 180, 0.18)",
            linecolor="#29465b",
        )
        fig.update_yaxes(
            gridcolor="rgba(120, 155, 180, 0.13)",
            zerolinecolor="rgba(120, 155, 180, 0.18)",
            linecolor="#29465b",
        )
    return fig


# ============================================================
# MAP GENERATOR
# ============================================================


def create_map(data: pd.DataFrame, height=620):

    if data is None or data.empty:
        return None

    required = {"latitude", "longitude", "sensor_family"}
    if not required.issubset(data.columns):
        return None

    map_data = data.copy()
    map_data["latitude"] = pd.to_numeric(
        map_data["latitude"], errors="coerce"
    )
    map_data["longitude"] = pd.to_numeric(
        map_data["longitude"], errors="coerce"
    )
    map_data = map_data.dropna(
        subset=["latitude", "longitude", "sensor_family"]
    )

    if map_data.empty:
        return None

    center = {
        "lat": float(map_data["latitude"].median()),
        "lon": float(map_data["longitude"].median()),
    }

    import math

    latitude_span = float(
        map_data["latitude"].max() - map_data["latitude"].min()
    )
    longitude_span = float(
        map_data["longitude"].max() - map_data["longitude"].min()
    ) * max(math.cos(math.radians(center["lat"])), 0.1)

    geographic_span = max(latitude_span, longitude_span, 0.05)
    zoom = max(1, min(12, math.log2(360 / geographic_span) - 1))

    hover_columns = [
        column
        for column in [
            "satellite_platform",
            "acq_datetime_utc",
            "frp_mw",
            "confidence_native",
            "confidence_scheme",
            "daynight",
            "source_product",
        ]
        if column in map_data.columns
    ]

    custom_columns = (
        ["observation_id"]
        if "observation_id" in map_data.columns
        else None
    )

    common_arguments = {
        "data_frame": map_data,
        "lat": "latitude",
        "lon": "longitude",
        "color": "sensor_family",
        "color_discrete_map": {
            "VIIRS": "#22D3EE",
            "MODIS": "#F5A742",
        },
        "category_orders": {
            "sensor_family": ["VIIRS", "MODIS"],
        },
        "hover_data": hover_columns,
        "custom_data": custom_columns,
        "center": center,
        "zoom": zoom,
        "height": height,
        "template": "plotly_dark",
    }

    if hasattr(px, "scatter_map"):
        fig = px.scatter_map(
            **common_arguments,
            map_style="carto-darkmatter",
        )
    else:
        fig = px.scatter_mapbox(
            **common_arguments,
            mapbox_style="carto-darkmatter",
        )

    fig.update_traces(
        marker={
            "size": 10,
            "opacity": 0.9,
        },
        selected=dict(
            marker=dict(size=13, opacity=1, color="#FFFFFF")
        ),
        unselected=dict(
            marker=dict(opacity=0.42)
        ),
    )

    fig.update_layout(
        margin={"r": 0, "t": 12, "l": 0, "b": 0},
        paper_bgcolor="#07111f",
        plot_bgcolor="#07111f",
        legend_title_text="Satellite sensor",
        legend=dict(
            bgcolor="rgba(7,17,31,0.86)",
            bordercolor="#244258",
            borderwidth=1,
            font=dict(color="#E2E8F0", size=12),
        ),
        hoverlabel=dict(
            bgcolor="#10283a",
            bordercolor="#3b7184",
            font=dict(color="#effaff"),
        ),
        uirevision="firesync-fire-map",
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

st.sidebar.markdown(
    f"""
    <div style="
        margin-top: 10px;
        padding: 12px;
        border: 1px solid #20384d;
        border-radius: 10px;
        background: rgba(13, 31, 47, .78);
        color: #a8bdce;
        font-size: 11px;
        line-height: 1.7;
    ">
        <div style="color:#55d6d0;font-weight:800;letter-spacing:1.2px;">
            ARCHIVE CONTEXT
        </div>
        <div><b style="color:#e8f1fb;">{len(df_all):,}</b> normalized records</div>
        <div>{min_date} → {max_date}</div>
        <div>MODIS + VIIRS · UTC timestamps</div>
    </div>
    """,
    unsafe_allow_html=True,
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
    """<div class="hero">
<div class="eyebrow">Sensor-aware satellite fire intelligence</div>
<h1>FIRESYNC</h1>
<div class="muted">
Two satellites. Different views. One clearer picture.
<br>
Harmonizing MODIS and VIIRS observations through transparent
evidence, candidate associations and uncertainty.
</div>
</div>""",
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

    st.markdown(
        '<div class="section-kicker">MISSION CONTROL  /  PROJECT OVERVIEW</div>',
        unsafe_allow_html=True,
    )

    # ========================================================
    # FIRESYNC 3D OBSERVATORY
    # ========================================================

    st.markdown(
        """
        <div style="
            padding: 8px 0 14px 0;
            color: #8ca9c5;
            letter-spacing: 3px;
            font-size: 11px;
            font-weight: 700;
        ">
            FIRESYNC / EARTH OBSERVATORY
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Use real coordinates from the currently filtered dataset.
    globe_data = (
        df[
            [
                "observation_id",
                "sensor_family",
                "satellite_platform",
                "latitude",
                "longitude",
                "acq_datetime_utc",
                "frp_mw",
                "confidence_class",
            ]
        ]
        .dropna(subset=["latitude", "longitude"])
        .copy()
    )

    # Keep the embedded component lightweight for the browser.
    if len(globe_data) > 3000:
        globe_data = globe_data.sample(
            n=3000,
            random_state=42,
        )

    globe_data["acq_datetime_utc"] = (
        globe_data["acq_datetime_utc"].astype(str)
    )

    observations_json = json.dumps(
        globe_data.to_dict(orient="records"),
        default=str,
    )

    globe_html = r"""
    <div id="firesync-shell">
      <div id="firesync-top">
        <div>
          <div class="eyebrow">SATELLITE FIRE INTELLIGENCE</div>
          <div class="heading">Earth Observatory</div>
          <div class="subheading">
            Explore satellite-observed thermal anomalies across Earth
          </div>
        </div>
        <div class="live-tag">
          <span class="live-dot"></span>
          HISTORICAL DATA
        </div>
      </div>

      <div id="globe-layout">
        <div id="globe-stage">
          <div id="loading">INITIALIZING 3D EARTH...</div>
          <div id="globe-label">EARTH / 3D VIEW</div>

          <div id="globe-controls">
            <button id="rotateBtn">PAUSE ROTATION</button>
            <button id="videoBtn">PAUSE VIDEO</button>
            <button id="resetBtn">RESET VIEW</button>
          </div>

          <div id="globe-legend">
            <span><i class="dot modis"></i> MODIS</span>
            <span><i class="dot viirs"></i> VIIRS</span>
          </div>
        </div>

        <aside id="info-panel">
          <div class="panel-label">OBSERVATION LAYERS</div>

          <label class="toggle-row">
            <span><i class="dot modis"></i> MODIS</span>
            <input type="checkbox" id="modisToggle" checked>
          </label>

          <label class="toggle-row">
            <span><i class="dot viirs"></i> VIIRS</span>
            <input type="checkbox" id="viirsToggle" checked>
          </label>

          <div class="panel-divider"></div>

          <div class="panel-label">DATASET OVERVIEW</div>
          <div class="data-number" id="totalCount">0</div>
          <div class="data-caption">Mapped observations</div>

          <div class="mini-stat">
            <span>MODIS</span>
            <strong id="modisCount">0</strong>
          </div>
          <div class="mini-stat">
            <span>VIIRS</span>
            <strong id="viirsCount">0</strong>
          </div>

          <div class="panel-divider"></div>

          <div class="panel-label">SELECTED OBSERVATION</div>
          <div id="selectedInfo">
            Select a marker on the globe to inspect its record.
          </div>

          <div class="panel-note">
            A satellite thermal anomaly is not necessarily a
            ground-confirmed fire. Marker positions represent
            archived observations, not confirmed fire boundaries.
          </div>
        </aside>
      </div>

      <div class="footer-note">
        NASA VIIRS visualization texture · FIRESYNC historical
        observations · Drag to rotate · Scroll to zoom
      </div>
    </div>

    <style>
      * { box-sizing: border-box; }

      body {
        margin: 0;
        background: #050b14;
        color: #edf5ff;
        font-family: Arial, sans-serif;
      }

      #firesync-shell {
        background:
          radial-gradient(ellipse at 35% 45%, #10283b, #050b14 72%);
        border: 1px solid #20384d;
        border-radius: 18px;
        padding: 22px;
        overflow: hidden;
      }

      #firesync-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 15px;
        margin-bottom: 18px;
      }

      .eyebrow, .panel-label {
        font-size: 10px;
        letter-spacing: 2px;
        color: #70c9db;
        font-weight: bold;
      }

      .heading {
        font-size: 27px;
        font-weight: 700;
        margin-top: 6px;
      }

      .subheading {
        color: #91a9bd;
        font-size: 12px;
        margin-top: 5px;
      }

      .live-tag {
        border: 1px solid #315366;
        border-radius: 30px;
        padding: 9px 12px;
        font-size: 9px;
        letter-spacing: 1px;
        color: #b6cbd9;
        white-space: nowrap;
      }

      .live-dot {
        display: inline-block;
        height: 7px;
        width: 7px;
        border-radius: 50%;
        background: #68c7c1;
        margin-right: 6px;
      }

      #globe-layout {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 235px;
        min-height: 540px;
        border: 1px solid #1c354a;
        border-radius: 13px;
        overflow: hidden;
      }

      #globe-stage {
        min-width: 0;
        min-height: 540px;
        position: relative;
        background: radial-gradient(circle, #10273a, #030812 75%);
      }

      #globe-stage canvas {
        width: 100%;
        height: 100%;
        display: block;
      }

      #loading {
        position: absolute;
        top: 48%;
        width: 100%;
        text-align: center;
        color: #76cbd9;
        font-size: 11px;
        letter-spacing: 2px;
        z-index: 2;
      }

      #globe-label {
        position: absolute;
        top: 16px;
        left: 18px;
        color: #91b4c8;
        font-size: 9px;
        letter-spacing: 2px;
        pointer-events: none;
      }

      #globe-controls {
        position: absolute;
        left: 14px;
        bottom: 15px;
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }

      #globe-controls button {
        background: #10263a;
        color: #d9edf8;
        border: 1px solid #34546a;
        border-radius: 6px;
        padding: 8px 10px;
        font-size: 9px;
        cursor: pointer;
      }

      #globe-controls button:hover {
        background: #1b4055;
      }

      #globe-legend {
        position: absolute;
        right: 15px;
        bottom: 18px;
        display: flex;
        gap: 13px;
        color: #c2d3df;
        font-size: 10px;
      }

      .dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 5px;
      }

      .modis { background: #ffb45e; }
      .viirs { background: #62e5e1; }

      #info-panel {
        background: rgba(8, 19, 32, 0.97);
        border-left: 1px solid #20384d;
        padding: 19px 15px;
      }

      .toggle-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: #d4e2ed;
        font-size: 12px;
        margin-top: 18px;
        cursor: pointer;
      }

      .toggle-row input {
        accent-color: #55d6d2;
      }

      .panel-divider {
        border-top: 1px solid #20364a;
        margin: 21px 0;
      }

      .data-number {
        font-size: 34px;
        font-weight: 700;
        margin-top: 13px;
        color: #e9f6ff;
      }

      .data-caption {
        font-size: 10px;
        color: #819caf;
        margin-top: 2px;
      }

      .mini-stat {
        display: flex;
        justify-content: space-between;
        margin-top: 15px;
        color: #9db2c5;
        font-size: 11px;
      }

      .mini-stat strong {
        color: #edf6ff;
      }

      #selectedInfo {
        color: #a8bdce;
        font-size: 11px;
        line-height: 1.8;
        margin-top: 12px;
        overflow-wrap: anywhere;
      }

      .panel-note {
        color: #7f98aa;
        font-size: 10px;
        line-height: 1.7;
        margin-top: 24px;
      }

      .footer-note {
        color: #718b9f;
        font-size: 10px;
        margin-top: 13px;
        line-height: 1.6;
      }

      @media(max-width: 650px) {
        #globe-layout {
          grid-template-columns: 1fr;
        }
        #info-panel {
          border-left: 0;
          border-top: 1px solid #20384d;
        }
        #globe-stage {
          min-height: 420px;
        }
        #globe-top {
          flex-wrap: wrap;
        }
      }
    </style>

    <script>
  window.__FIRESYNC_OBSERVATIONS__ = __OBSERVATIONS__;
</script>

<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js"
  }
}
</script>

<script type="module">
  import * as THREE from
    "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";

  import { OrbitControls } from
    "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js";

      const observations = window.__FIRESYNC_OBSERVATIONS__ || [];

      const stage = document.getElementById("globe-stage");
      const loading = document.getElementById("loading");

      const scene = new THREE.Scene();

      const camera = new THREE.PerspectiveCamera(
        42,
        stage.clientWidth / stage.clientHeight,
        0.1,
        100
      );

      camera.position.set(0, 0, 2.8);

      const renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true
      });

      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setSize(stage.clientWidth, stage.clientHeight);
      renderer.outputColorSpace = THREE.SRGBColorSpace;

      stage.appendChild(renderer.domElement);

      const controls = new OrbitControls(
        camera,
        renderer.domElement
      );

      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.minDistance = 1.4;
      controls.maxDistance = 5;
      controls.autoRotate = true;
      controls.autoRotateSpeed = 0.45;

      scene.add(new THREE.AmbientLight(0xffffff, 1.3));

      const sun = new THREE.DirectionalLight(0xffffff, 2.0);
      sun.position.set(4, 2, 5);
      scene.add(sun);

      // Background stars.
      const starPositions = [];

      for (let i = 0; i < 1400; i++) {
        starPositions.push(
          (Math.random() - 0.5) * 16,
          (Math.random() - 0.5) * 16,
          (Math.random() - 0.5) * 16
        );
      }

      const starGeometry = new THREE.BufferGeometry();

      starGeometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(starPositions, 3)
      );

      const stars = new THREE.Points(
        starGeometry,
        new THREE.PointsMaterial({
          color: 0x9bbbd2,
          size: 0.012
        })
      );

      scene.add(stars);

      const earthGroup = new THREE.Group();
      scene.add(earthGroup);

      const video = document.createElement("video");

      video.src =
        "/app/static/fires_frp_VIIRS_eq_nl_2048p30.mp4";

      video.crossOrigin = "anonymous";
      video.loop = true;
      video.muted = true;
      video.autoplay = true;
      video.playsInline = true;
      video.preload = "auto";

      const earthGeometry = new THREE.SphereGeometry(
        1,
        96,
        64
      );

      const earthMaterial = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        roughness: 1,
        metalness: 0
      });

      const earth = new THREE.Mesh(
        earthGeometry,
        earthMaterial
      );

      // Invert sphere faces to display the video texture inside
      // the sphere geometry's normal viewing surface.
      earthGroup.add(earth);

      const videoTexture = new THREE.VideoTexture(video);

      videoTexture.colorSpace = THREE.SRGBColorSpace;

      earth.material.map = videoTexture;
      earth.material.needsUpdate = true;

      // Latitude/longitude to spherical coordinates.
      function geographicPosition(lat, lon, radius) {
        const phi = (90 - lat) * Math.PI / 180;
        const theta = (lon + 180) * Math.PI / 180;

        return new THREE.Vector3(
          -radius * Math.sin(phi) * Math.cos(theta),
          radius * Math.cos(phi),
          radius * Math.sin(phi) * Math.sin(theta)
        );
      }

      const sensorGroups = {
        MODIS: new THREE.Group(),
        VIIRS: new THREE.Group()
      };

      earthGroup.add(sensorGroups.MODIS);
      earthGroup.add(sensorGroups.VIIRS);

      const markerRecords = {
        MODIS: [],
        VIIRS: []
      };

      for (const sensor of ["MODIS", "VIIRS"]) {
        const records = observations.filter(
          row => row.sensor_family === sensor
        );

        const positions = [];

        records.forEach((row, index) => {
          const position = geographicPosition(
            Number(row.latitude),
            Number(row.longitude),
            1.014
          );

          positions.push(
            position.x,
            position.y,
            position.z
          );

          markerRecords[sensor].push(row);
        });

        const geometry = new THREE.BufferGeometry();

        geometry.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(positions, 3)
        );

        const points = new THREE.Points(
          geometry,
          new THREE.PointsMaterial({
            color: sensor === "MODIS" ? 0xffb45e : 0x62e5e1,
            size: 0.025,
            sizeAttenuation: true,
            transparent: true,
            opacity: 0.95
          })
        );

        points.userData.sensor = sensor;
        sensorGroups[sensor].add(points);
      }

      document.getElementById("totalCount").textContent =
        observations.length.toLocaleString();

      document.getElementById("modisCount").textContent =
        markerRecords.MODIS.length.toLocaleString();

      document.getElementById("viirsCount").textContent =
        markerRecords.VIIRS.length.toLocaleString();

      document.getElementById("modisToggle").addEventListener(
        "change",
        event => {
          sensorGroups.MODIS.visible = event.target.checked;
        }
      );

      document.getElementById("viirsToggle").addEventListener(
        "change",
        event => {
          sensorGroups.VIIRS.visible = event.target.checked;
        }
      );

      document.getElementById("rotateBtn").addEventListener(
        "click",
        event => {
          controls.autoRotate = !controls.autoRotate;
          event.target.textContent = controls.autoRotate
            ? "PAUSE ROTATION"
            : "START ROTATION";
        }
      );

      document.getElementById("videoBtn").addEventListener(
        "click",
        event => {
          if (video.paused) {
            video.play();
            event.target.textContent = "PAUSE VIDEO";
          } else {
            video.pause();
            event.target.textContent = "PLAY VIDEO";
          }
        }
      );

      document.getElementById("resetBtn").addEventListener(
        "click",
        () => {
          camera.position.set(0, 0, 2.8);
          controls.target.set(0, 0, 0);
          controls.update();
        }
      );

      // Select nearby observation markers.
      const raycaster = new THREE.Raycaster();
      raycaster.params.Points.threshold = 0.035;

      const pointer = new THREE.Vector2();

      renderer.domElement.addEventListener(
        "click",
        event => {
          const rect = renderer.domElement.getBoundingClientRect();

          pointer.x =
            ((event.clientX - rect.left) / rect.width) * 2 - 1;

          pointer.y =
            -((event.clientY - rect.top) / rect.height) * 2 + 1;

          raycaster.setFromCamera(pointer, camera);

          const visibleGroups = [
            sensorGroups.MODIS,
            sensorGroups.VIIRS
          ].filter(group => group.visible);

          const targets = [];

          visibleGroups.forEach(group => {
            group.children.forEach(child => targets.push(child));
          });

          const hits = raycaster.intersectObjects(targets);

          if (!hits.length) return;

          const point = hits[0];
          const sensor = point.object.userData.sensor;
          const record = markerRecords[sensor][point.index];

          const details = [
            ["Sensor", record.sensor_family],
            ["Platform", record.satellite_platform],
            ["Latitude", Number(record.latitude).toFixed(4)],
            ["Longitude", Number(record.longitude).toFixed(4)],
            ["Acquisition UTC", record.acq_datetime_utc],
            ["FRP (MW)", record.frp_mw ?? "Not available"],
            ["Confidence", record.confidence_class ?? "Not available"],
            ["Observation ID", record.observation_id]
          ];

          document.getElementById("selectedInfo").innerHTML =
            details.map(([key, value]) =>
              `<div><b>${key}:</b> ${String(value)}</div>`
            ).join("");
        }
      );

      // Start video texture when the browser permits autoplay.
      video.play().catch(() => {
        document.getElementById("videoBtn").textContent =
          "PLAY VIDEO";
      });

      video.addEventListener("loadeddata", () => {
        loading.style.display = "none";
      });

      video.addEventListener("error", () => {
        loading.textContent =
          "NASA VIDEO UNAVAILABLE — CHECK STATIC FILE ACCESS";
      });

      // Responsive canvas.
      const resizeObserver = new ResizeObserver(() => {
        const width = stage.clientWidth;
        const height = stage.clientHeight;

        camera.aspect = width / height;
        camera.updateProjectionMatrix();

        renderer.setSize(width, height);
      });

      resizeObserver.observe(stage);

      function animate() {
        requestAnimationFrame(animate);

        controls.update();
        renderer.render(scene, camera);
      }

      animate();
    </script>
    """

    globe_html = globe_html.replace(
        "__OBSERVATIONS__",
        observations_json,
    )

    components.html(
        globe_html,
        height=750,
        scrolling=False,
    )

    st.divider()


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
            color_discrete_map={"VIIRS": "#22D3EE", "MODIS": "#F5A742"},
            barmode="group",
            template="plotly_dark",
        )

        fig.update_layout(
            paper_bgcolor="#07111f",
            plot_bgcolor="#07111f",
            xaxis_title="Satellite platform",
        )

        fig = style_figure(fig)
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
            color_discrete_map={"VIIRS": "#22D3EE", "MODIS": "#F5A742"},
            barmode="group",
            template="plotly_dark",
        )

        fig.update_layout(
            paper_bgcolor="#07111f",
            plot_bgcolor="#07111f",
            xaxis_title="UTC year",
        )

        fig = style_figure(fig)
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

    fig = style_figure(fig)
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

    render_module_header(
        "MODULE 02  /  GEOSPATIAL EXPLORATION",
        "Fire Map Explorer",
        "Explore archived MODIS and VIIRS thermal-anomaly observations, "
        "inspect individual records, and investigate selected map regions.",
    )

    st.warning(
        "Each marker is an individual satellite observation. Nearby records "
        "may represent repeated detections or different overpasses; they are "
        "not automatically unique fires or confirmed fire perimeters."
    )

    # Stable slider bounds use the complete loaded archive. The current
    # global filters and local controls are applied after the widgets.
    map_df = df.copy()

    geo_source = df_all.dropna(subset=["latitude", "longitude"])
    lat_min = float(geo_source["latitude"].min())
    lat_max = float(geo_source["latitude"].max())
    lon_min = float(geo_source["longitude"].min())
    lon_max = float(geo_source["longitude"].max())

    frp_source = pd.to_numeric(df_all["frp_mw"], errors="coerce")
    frp_max = float(frp_source.max()) if frp_source.notna().any() else 1.0
    frp_max = max(frp_max, 1.0)

    with st.expander("MAP FILTERS  /  AREA, INTENSITY & SENSOR CONFIDENCE", expanded=True):

        geo_col, signal_col = st.columns([1.05, 0.95], gap="large")

        with geo_col:
            st.markdown("##### Geographic extent")
            latitude_range = st.slider(
                "Latitude bounds",
                min_value=lat_min,
                max_value=lat_max,
                value=(lat_min, lat_max),
                step=max((lat_max - lat_min) / 1000, 0.00001),
                format="%.5f",
                key="firemap_latitude_range_v2",
            )
            longitude_range = st.slider(
                "Longitude bounds",
                min_value=lon_min,
                max_value=lon_max,
                value=(lon_min, lon_max),
                step=max((lon_max - lon_min) / 1000, 0.00001),
                format="%.5f",
                key="firemap_longitude_range_v2",
            )
            st.caption(
                "Bounds are based on the geographic extent of the loaded archive, "
                "not an official protected-area boundary."
            )

        with signal_col:
            st.markdown("##### Observation characteristics")
            frp_range = st.slider(
                "Fire radiative power (MW)",
                min_value=0.0,
                max_value=frp_max,
                value=(0.0, frp_max),
                step=max(frp_max / 1000, 0.01),
                key="firemap_frp_range_v2",
            )

            daynight_options = [
                value
                for value in ["D", "N"]
                if value in df_all["daynight"].dropna().astype(str).str.upper().unique()
            ]
            selected_daynight = st.multiselect(
                "Acquisition illumination",
                options=daynight_options,
                default=daynight_options,
                format_func=lambda value: {"D": "Day", "N": "Night"}.get(value, value),
                key="firemap_daynight_v2",
            )

            include_missing_values = st.checkbox(
                "Retain observations with missing filter values",
                value=True,
                key="firemap_include_missing_v2",
                help=(
                    "When enabled, records without a value for a selected "
                    "attribute are retained instead of silently discarded."
                ),
            )

        st.divider()
        st.markdown("##### Native sensor confidence")

        conf_left, conf_right = st.columns(2, gap="large")

        viirs_options = sorted(
            df_all.loc[
                df_all["sensor_family"] == "VIIRS",
                "confidence_native",
            ]
            .dropna()
            .astype(str)
            .str.lower()
            .unique()
            .tolist()
        )

        with conf_left:
            st.markdown("**VIIRS · categorical**")
            selected_viirs_confidence = st.multiselect(
                "Native confidence classes",
                options=viirs_options,
                default=viirs_options,
                format_func=lambda value: {
                    "l": "Low (l)",
                    "n": "Nominal (n)",
                    "h": "High (h)",
                }.get(value, value),
                key="firemap_viirs_confidence_v2",
            )
            st.caption("VIIRS classes are preserved as low / nominal / high.")

        with conf_right:
            st.markdown("**MODIS · numeric**")
            modis_confidence_range = st.slider(
                "Native numeric confidence (0–100)",
                min_value=0.0,
                max_value=100.0,
                value=(0.0, 100.0),
                step=1.0,
                key="firemap_modis_confidence_v2",
            )
            st.caption("MODIS numeric confidence is not interpreted as a fire probability.")

    # --------------------------------------------------------
    # APPLY LOCAL FILTERS
    # --------------------------------------------------------

    latitude_values = pd.to_numeric(map_df["latitude"], errors="coerce")
    longitude_values = pd.to_numeric(map_df["longitude"], errors="coerce")
    frp_values = pd.to_numeric(map_df["frp_mw"], errors="coerce")
    family_values = map_df["sensor_family"].astype(str)

    daynight_values = (
        map_df["daynight"].astype("string").str.upper()
        if "daynight" in map_df.columns
        else pd.Series(pd.NA, index=map_df.index, dtype="string")
    )
    viirs_confidence_values = (
        map_df["confidence_native"].astype("string").str.lower()
        if "confidence_native" in map_df.columns
        else pd.Series(pd.NA, index=map_df.index, dtype="string")
    )
    modis_confidence_values = pd.to_numeric(
        map_df["confidence_numeric"], errors="coerce"
    )

    missing_pass = include_missing_values

    geographic_mask = (
        latitude_values.between(*latitude_range)
        & longitude_values.between(*longitude_range)
    )
    frp_mask = frp_values.between(*frp_range) | (
        frp_values.isna() & missing_pass
    )
    daynight_mask = daynight_values.isin(selected_daynight) | (
        daynight_values.isna() & missing_pass
    )
    viirs_mask = (
        family_values.ne("VIIRS")
        | viirs_confidence_values.isin(selected_viirs_confidence)
        | (viirs_confidence_values.isna() & missing_pass)
    )
    modis_mask = (
        family_values.ne("MODIS")
        | modis_confidence_values.between(*modis_confidence_range)
        | (modis_confidence_values.isna() & missing_pass)
    )

    filtered_map_df = map_df.loc[
        geographic_mask
        & frp_mask
        & daynight_mask
        & viirs_mask
        & modis_mask
    ].copy()

    # Render every record for the current archive. A cap is only offered
    # when future datasets grow beyond 5,000 records.
    map_display_df = filtered_map_df
    if len(filtered_map_df) > 5000:
        point_limit = st.slider(
            "Map rendering limit",
            min_value=1000,
            max_value=min(10000, len(filtered_map_df)),
            value=5000,
            step=500,
            key="firemap_point_limit_v2",
            help="The observation register retains the full filtered dataset.",
        )
        map_display_df = filtered_map_df.sample(
            n=point_limit,
            random_state=42,
        )

    total_filtered = len(filtered_map_df)
    modis_filtered = int((filtered_map_df["sensor_family"] == "MODIS").sum())
    viirs_filtered = int((filtered_map_df["sensor_family"] == "VIIRS").sum())
    median_frp = pd.to_numeric(
        filtered_map_df["frp_mw"], errors="coerce"
    ).median()

    metric_cols = st.columns(4)
    metric_cols[0].metric("Filtered observations", f"{total_filtered:,}")
    metric_cols[1].metric("VIIRS records", f"{viirs_filtered:,}")
    metric_cols[2].metric("MODIS records", f"{modis_filtered:,}")
    metric_cols[3].metric(
        "Median FRP",
        f"{median_frp:.2f} MW" if pd.notna(median_frp) else "—",
    )

    st.markdown("##### Regional observation map")
    st.caption(
        "VIIRS is shown in cyan; MODIS is shown in amber. "
        "Use the map toolbar to zoom, pan, select individual points, "
        "or draw a rectangle / lasso around a group of observations."
    )

    map_event = None
    selected_ids = []
    selected_observations = pd.DataFrame()

    if filtered_map_df.empty:
        st.info(
            "No observations match the current global and local filters. "
            "Widen the geographic, FRP, day/night or confidence selections."
        )
    else:
        fig = create_map(map_display_df, height=650)

        if fig is not None:
            map_event = st.plotly_chart(
                fig,
                use_container_width=True,
                key="firesync_fire_map_selection_v2",
                on_select="rerun",
                selection_mode=("points", "box", "lasso"),
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                    "modeBarButtonsToAdd": ["select2d", "lasso2d"],
                },
            )

            selection_data = map_event.get("selection", {})
            selected_points = selection_data.get("points", [])

            for point in selected_points:
                custom_data = point.get("customdata")
                if isinstance(custom_data, (list, tuple)) and custom_data:
                    selected_ids.append(str(custom_data[0]))

            selected_ids = list(dict.fromkeys(selected_ids))

            if selected_ids:
                selected_observations = filtered_map_df.loc[
                    filtered_map_df["observation_id"].astype(str).isin(selected_ids)
                ].copy()

    # --------------------------------------------------------
    # SELECTION INVESTIGATION + PRIMARY RECORD INSPECTOR
    # --------------------------------------------------------

    if not selected_observations.empty:
        st.markdown("##### Selection investigation")

        selection_counts = selected_observations["sensor_family"].value_counts()
        selection_frp = pd.to_numeric(
            selected_observations["frp_mw"], errors="coerce"
        )
        selection_metrics = st.columns(4)
        selection_metrics[0].metric(
            "Selected observations", f"{len(selected_observations):,}"
        )
        selection_metrics[1].metric(
            "VIIRS selected", f"{selection_counts.get('VIIRS', 0):,}"
        )
        selection_metrics[2].metric(
            "MODIS selected", f"{selection_counts.get('MODIS', 0):,}"
        )
        selection_median = selection_frp.median()
        selection_metrics[3].metric(
            "Selection median FRP",
            f"{selection_median:.2f} MW"
            if pd.notna(selection_median)
            else "—",
        )

        primary_record = selected_observations.iloc[0]
        with st.expander("Primary selected observation · inspect record", expanded=True):
            detail_cols = st.columns(4)
            detail_cols[0].metric(
                "Sensor", str(primary_record.get("sensor_family", "—"))
            )
            detail_cols[1].metric(
                "Platform", str(primary_record.get("satellite_platform", "—"))
            )
            primary_frp = primary_record.get("frp_mw")
            detail_cols[2].metric(
                "FRP",
                f"{float(primary_frp):.2f} MW"
                if pd.notna(primary_frp)
                else "—",
            )
            detail_cols[3].metric(
                "Native confidence",
                str(primary_record.get("confidence_native", "—")),
            )

            detail_fields = [
                "observation_id",
                "acq_datetime_utc",
                "latitude",
                "longitude",
                "confidence_scheme",
                "daynight",
                "source_product",
                "scan_native",
                "track_native",
            ]
            detail_table = pd.DataFrame(
                [
                    {
                        "Field": field,
                        "Value": str(primary_record.get(field, "—")),
                    }
                    for field in detail_fields
                    if field in selected_observations.columns
                ]
            )
            show_table(detail_table, 280)

        st.download_button(
            "Download selected observations · CSV",
            data=selected_observations.to_csv(index=False),
            file_name="firesync_selected_observations.csv",
            mime="text/csv",
            key="firemap_download_selected_v2",
        )
        st.caption(
            "A selected group is a set of mapped detections, not a confirmed "
            "single fire, fire perimeter or physical event."
        )
    else:
        st.info(
            "Select one or more map points with the point, box or lasso tool "
            "to open the observation inspector and selection summary."
        )

    # --------------------------------------------------------
    # FULL FILTERED OBSERVATION REGISTER
    # --------------------------------------------------------

    st.divider()
    st.markdown("##### Filtered observation register")

    if not filtered_map_df.empty:
        register = filtered_map_df[
            observation_columns(filtered_map_df)
        ].sort_values(
            "acq_datetime_utc",
            ascending=False,
        )

        st.download_button(
            "Download all filtered observations · CSV",
            data=register.to_csv(index=False),
            file_name="firesync_filtered_observations.csv",
            mime="text/csv",
            key="firemap_download_filtered_v2",
        )
        show_table(register, 430)
    else:
        st.info("The observation register is empty for the current filters.")

    st.caption(
        f"Showing {len(map_display_df):,} points on the map from "
        f"{len(filtered_map_df):,} locally filtered records. "
        "The table and CSV retain the complete filtered set."
    )


# ============================================================
# MODULE 3 — ACTIVITY CALENDAR
# ============================================================

elif page == "Activity Calendar":

    render_module_header(
        "MODULE 03  /  TEMPORAL PATTERNS",
        "Historical Activity Calendar",
        "Explore daily and monthly retrieval patterns by sensor family. "
        "Observed counts describe the archive, not ground-truth fire occurrence.",
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

    total_calendar_records = len(calendar_df)
    active_days = int(calendar_df["date"].nunique())
    daily_totals = daily.groupby("date")["observations"].sum()
    peak_day = daily_totals.idxmax() if not daily_totals.empty else None
    calendar_median_frp = pd.to_numeric(
        calendar_df["frp_mw"], errors="coerce"
    ).median()

    calendar_metrics = st.columns(4)
    calendar_metrics[0].metric(
        "Archived observations", f"{total_calendar_records:,}"
    )
    calendar_metrics[1].metric(
        "Dates with records", f"{active_days:,}"
    )
    calendar_metrics[2].metric(
        "Highest retrieval day",
        str(peak_day) if peak_day is not None else "—",
    )
    calendar_metrics[3].metric(
        "Median FRP",
        f"{calendar_median_frp:.2f} MW"
        if pd.notna(calendar_median_frp)
        else "—",
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
            color_discrete_map={"VIIRS": "#22D3EE", "MODIS": "#F5A742"},
            markers=True,
            template="plotly_dark",
        )

        fig.update_layout(
            paper_bgcolor="#07111f",
            plot_bgcolor="#07111f",
            xaxis_title="Acquisition date (UTC)",
            yaxis_title="Retrieved observations",
        )

        fig = style_figure(fig)
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
            color_discrete_map={"VIIRS": "#22D3EE", "MODIS": "#F5A742"},
            barmode="group",
            template="plotly_dark",
        )

        fig.update_layout(
            paper_bgcolor="#07111f",
            plot_bgcolor="#07111f",
        )

        fig = style_figure(fig)
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

    render_module_header(
        "MODULE 04  /  CROSS-SENSOR RELATIONSHIPS",
        "MODIS–VIIRS Association Explorer",
        "Investigate spatially and temporally constrained candidate links. "
        "Associations remain hypotheses, not confirmed same-fire determinations.",
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

            fig = style_figure(fig)
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

    render_module_header(
        "MODULE 05  /  SENSOR COMPARISON",
        "Harmonization Lab",
        "Compare native confidence, measurement channels and association sensitivity "
        "without forcing unlike sensor measurements into a shared scale.",
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

    sensitivity_signature = (
        len(df),
        str(start_date),
        str(end_date),
        tuple(selected_platforms),
        tuple(selected_sensors),
        float(sensitivity_time),
        tuple(sorted(thresholds)),
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
            st.session_state["sensitivity_signature"] = (
                sensitivity_signature
            )

    if (
        "sensitivity_results" in st.session_state
        and st.session_state.get("sensitivity_signature")
        == sensitivity_signature
    ):

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
            style_figure(fig),
            use_container_width=True,
        )

    elif "sensitivity_results" in st.session_state:
        st.info(
            "Sensitivity inputs changed. Recalculate to refresh these results "
            "for the current global filters and thresholds."
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

    benchmark_signature = (
        len(df),
        str(start_date),
        str(end_date),
        tuple(selected_platforms),
        tuple(selected_sensors),
        float(benchmark_distance),
        float(benchmark_time),
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
            st.session_state["benchmark_signature"] = (
                benchmark_signature
            )

        except Exception as exc:

            st.error(
                f"Benchmark failed: {exc}"
            )

    if (
        "benchmark_result" in st.session_state
        and st.session_state.get("benchmark_signature")
        == benchmark_signature
    ):

        benchmark_result = st.session_state[
            "benchmark_result"
        ]

        benchmark_metrics = st.columns(4)
        benchmark_metrics[0].metric(
            "Reference runtime",
            f"{benchmark_result['Reference runtime (seconds)']:.3f} s",
        )
        benchmark_metrics[1].metric(
            "Optimized runtime",
            f"{benchmark_result['Optimized runtime (seconds)']:.3f} s",
        )
        benchmark_metrics[2].metric(
            "Measured speedup",
            f"{benchmark_result['Measured speedup']:.2f}×"
            if pd.notna(benchmark_result["Measured speedup"])
            else "—",
        )
        benchmark_metrics[3].metric(
            "Candidate edges",
            f"{benchmark_result['Reference edges']:,}",
        )

        show_table(
            pd.DataFrame(
                [
                    {"Validation check": "Candidate edge IDs", "Identical": benchmark_result["Candidate edge IDs identical"]},
                    {"Validation check": "Cluster memberships", "Identical": benchmark_result["Cluster memberships identical"]},
                    {"Validation check": "Unmatched observation IDs", "Identical": benchmark_result["Unmatched IDs identical"]},
                ]
            ),
            180,
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

    elif "benchmark_result" in st.session_state:
        st.info(
            "Benchmark inputs or global filters changed. Run the comparison "
            "again to refresh the evidence for the current configuration."
        )

    st.warning(
        "Sensitivity and benchmark outputs describe algorithm behavior "
        "under selected conditions. They do not establish an optimal "
        "scientific threshold or confirm physical fire events."
    )

    st.divider()

    st.markdown(
        "### 🔬 Statistical Sensor Harmonization & Calibration Engine"
    )
    st.caption(
        "Addresses PRD FR-05: Calibrates sensor differences across overlapping observation "
        "periods. Eliminates the 10×–20× raw detection inflation caused by VIIRS's finer 375 m "
        "footprint without destroying physical fire activity signals, producing comparable "
        "estimates with quantified 95% uncertainty intervals."
    )

    # 1. Controls
    cal_col1, cal_col2 = st.columns([2, 1])
    with cal_col1:
        model_choice = st.selectbox(
            "Calibration Model",
            [
                "Two-Stage Hurdle Model",
                "Random Forest Harmonizer",
                "Robust Log-Log Huber (Baseline)",
                "Empirical Area Ratio (Baseline)",
            ],
            index=0,
            key="harmonization_model_choice",
            help="Select the statistical calibration model used to generate harmonized estimates.",
        )
    with cal_col2:
        val_split = st.selectbox(
            "Validation Holdout Year",
            [2024, 2023, 2022],
            index=0,
            key="harmonization_val_split",
            help="Observations from this year onward are held out as an independent test set.",
        )

    # 2. Run / Cache Model Training
    with st.spinner("Calculating sensor calibration and evaluating models..."):
        try:
            source_data = df if len(df) >= 10 else df_all
            analytical_units = build_daily_analytical_units(source_data)
            metrics_table, trained_models = evaluate_harmonization_models(
                analytical_units, split_year=val_split
            )
            harmonized_df = generate_harmonized_dataset(
                source_data, preferred_model_name=model_choice
            )
        except Exception as exc:
            st.error(f"Harmonization engine error: {exc}")
            analytical_units = None
            metrics_table = None
            harmonized_df = None

    if metrics_table is not None and harmonized_df is not None:
        # 3. Model Benchmark & Evaluation Table
        st.markdown("#### 1. Independent Validation & Model Comparison")
        st.caption(
            f"Evaluated on independent chronological holdout: Training (year < {val_split}) vs. "
            f"Holdout Validation (year ≥ {val_split}). Metrics show predictive accuracy for MODIS-equivalent activity."
        )
        show_table(metrics_table, 230)

        # 4. Summary Metrics Cards
        cal_summary = st.columns(4)
        total_viirs = harmonized_df["raw_viirs_count"].sum()
        total_modis = harmonized_df["raw_modis_count"].sum()
        total_calibrated = harmonized_df["calibrated_modis_equivalent"].sum()
        disparity_reduction = (
            ((total_viirs - total_calibrated) / total_viirs * 100.0)
            if total_viirs > 0
            else 0.0
        )

        cal_summary[0].metric(
            "Raw VIIRS Detections",
            f"{int(total_viirs):,}",
            help="Uncalibrated total VIIRS hotspots (375 m resolution).",
        )
        cal_summary[1].metric(
            "Raw MODIS Detections",
            f"{int(total_modis):,}",
            help="Native MODIS hotspots (1 km resolution).",
        )
        cal_summary[2].metric(
            "Calibrated Estimate",
            f"{total_calibrated:.1f}",
            help="Harmonized MODIS-equivalent detection count.",
        )
        cal_summary[3].metric(
            "Observational Bias Corrected",
            f"{disparity_reduction:.1f}%",
            help="Percentage of raw count inflation resolved by calibration.",
        )

        # 5. Interactive Raw vs. Harmonized Fire Trends
        st.divider()
        st.markdown("#### 2. Raw vs. Harmonized Fire Activity Time-Series")
        st.caption(
            "Notice how raw VIIRS counts surge due to higher spatial resolution (375 m vs 1 km), "
            "whereas the Calibrated MODIS-Equivalent curve (with shaded 95% Confidence Band) "
            "normalizes this disparity and enables valid historical comparisons."
        )

        trend_view = st.radio(
            "Temporal Roll-up",
            ["Monthly Aggregate", "Daily Observations"],
            horizontal=True,
            key="harmonization_trend_view",
        )

        if trend_view == "Monthly Aggregate":
            harmonized_df["year_month"] = (
                pd.to_datetime(harmonized_df["acq_date"]).dt.to_period("M").dt.to_timestamp()
            )
            plot_df = (
                harmonized_df.groupby("year_month")
                .agg(
                    raw_viirs=("raw_viirs_count", "sum"),
                    raw_modis=("raw_modis_count", "sum"),
                    calibrated=("calibrated_modis_equivalent", "sum"),
                    ci_lower=("ci_lower_95", "sum"),
                    ci_upper=("ci_upper_95", "sum"),
                )
                .reset_index()
            )
            x_col = "year_month"
            x_title = "Month"
        else:
            plot_df = harmonized_df.sort_values("acq_date")
            x_col = "acq_date"
            x_title = "Acquisition Date"

        fig_trend = go.Figure()

        # Shaded 95% Confidence Interval band
        fig_trend.add_trace(
            go.Scatter(
                x=plot_df[x_col],
                y=plot_df["ci_upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig_trend.add_trace(
            go.Scatter(
                x=plot_df[x_col],
                y=plot_df["ci_lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(34, 211, 238, 0.15)",
                name="95% Confidence Band",
                hoverinfo="skip",
            )
        )

        # Raw VIIRS
        fig_trend.add_trace(
            go.Scatter(
                x=plot_df[x_col],
                y=plot_df["raw_viirs"],
                mode="lines+markers",
                name="Raw VIIRS (375m)",
                line=dict(color="#f5a742", width=2, dash="dot"),
                marker=dict(size=5, color="#f5a742"),
            )
        )

        # Raw MODIS
        fig_trend.add_trace(
            go.Scatter(
                x=plot_df[x_col],
                y=plot_df["raw_modis"],
                mode="lines+markers",
                name="Raw MODIS (1km)",
                line=dict(color="#e11d48", width=2),
                marker=dict(size=6, color="#e11d48"),
            )
        )

        # Calibrated Estimate
        fig_trend.add_trace(
            go.Scatter(
                x=plot_df[x_col],
                y=plot_df["calibrated"],
                mode="lines+markers",
                name="Calibrated MODIS-Equivalent (Harmonized)",
                line=dict(color="#22d3ee", width=3),
                marker=dict(size=6, color="#22d3ee"),
            )
        )

        fig_trend.update_layout(
            title=f"Fire Activity Trends: Raw vs. Harmonized ({model_choice})",
            xaxis_title=x_title,
            yaxis_title="Observation Count",
            hovermode="x unified",
        )
        st.plotly_chart(style_figure(fig_trend, height=450), use_container_width=True)

        # 6. Data Quality & Attribution Distribution
        st.divider()
        qual_col1, qual_col2 = st.columns([1, 1])

        with qual_col1:
            st.markdown("#### 3. Data Quality Attribution")
            st.caption("Every record is tagged to preserve observation provenance.")
            flag_counts = (
                harmonized_df["data_quality_flag"]
                .value_counts()
                .reset_index()
            )
            flag_counts.columns = ["Data Quality Flag", "Days"]
            show_table(flag_counts, 220)

        with qual_col2:
            st.markdown("#### 4. Export Harmonized Dataset")
            st.caption(
                "Export calibrated daily records with raw observations, uncertainty bands, "
                "and provenance flags for reproducible downstream research."
            )
            export_csv = harmonized_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Harmonized Fire Activity Dataset · CSV",
                data=export_csv,
                file_name=f"firesync_harmonized_dataset_{model_choice.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                key="download_harmonized_csv_btn",
                type="primary",
            )


# ============================================================
# MODULE 6 — ACTIVITY INVESTIGATION
# ============================================================

elif page == "Activity Investigation":

    render_module_header(
        "MODULE 06  /  RECORD-LEVEL INVESTIGATION",
        "Activity Investigation",
        "Trace an observation by its unique identifier or inspect the members "
        "and internal candidate links of a generated association cluster.",
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

                fig = style_figure(fig)
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

                fig = style_figure(fig)
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

    render_module_header(
        "MODULE 07  /  PROVENANCE & SCIENTIFIC SAFEGUARDS",
        "Evidence Centre",
        "Review the normalized archive, processing provenance, benchmark evidence, "
        "methodological choices and interpretation limits.",
    )

    provenance_tab, coverage_tab, benchmark_tab, methods_tab, schema_tab = st.tabs(
        [
            "Dataset provenance",
            "Platform coverage",
            "Engine evidence",
            "Methodology & limits",
            "Data schema",
        ]
    )

    with provenance_tab:
        st.markdown("##### Dataset identity")

        provenance = {
            "Dataset": DATA_FILE.name,
            "Records": f"{len(df_all):,}",
            "Columns": len(df_all.columns),
            "First UTC timestamp": str(df_all["acq_datetime_utc"].min()),
            "Last UTC timestamp": str(df_all["acq_datetime_utc"].max()),
            "Source products": (
                ", ".join(
                    sorted(
                        df_all["source_product"].dropna().astype(str).unique()
                    )
                )
                if "source_product" in df_all.columns
                else "Not available"
            ),
            "Sensor families": ", ".join(
                sorted(df_all["sensor_family"].dropna().astype(str).unique())
            ),
        }

        show_table(
            pd.DataFrame(
                [{"Field": key, "Value": value} for key, value in provenance.items()]
            ),
            330,
        )

        st.download_button(
            "Download normalized dataset manifest · CSV",
            data=pd.DataFrame(
                [{"Field": key, "Value": value} for key, value in provenance.items()]
            ).to_csv(index=False),
            file_name="firesync_dataset_manifest.csv",
            mime="text/csv",
            key="evidence_manifest_download",
        )

    with coverage_tab:
        st.markdown("##### Platform coverage in normalized archive")

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

        coverage_metrics = st.columns(3)
        coverage_metrics[0].metric(
            "Sensor families",
            f"{df_all['sensor_family'].nunique():,}",
        )
        coverage_metrics[1].metric(
            "Platforms",
            f"{df_all['satellite_platform'].nunique():,}",
        )
        coverage_metrics[2].metric(
            "Source products",
            f"{df_all['source_product'].nunique():,}"
            if "source_product" in df_all.columns
            else "—",
        )

        show_table(platform_summary, 360)

    with benchmark_tab:
        st.markdown("##### Historical engine benchmark evidence")

        report = safe_read_json(ENGINE_REPORT)

        if report is not None:
            if isinstance(report, dict) and "read_error" in report:
                st.warning(f"Benchmark report could not be read: {report['read_error']}")
            else:
                st.json(report, expanded=False)

        elif ENGINE_CSV.exists():
            try:
                benchmark = pd.read_csv(ENGINE_CSV)
                show_table(benchmark, 380)
                st.download_button(
                    "Download benchmark evidence · CSV",
                    data=benchmark.to_csv(index=False),
                    file_name="firesync_engine_benchmark_evidence.csv",
                    mime="text/csv",
                    key="evidence_benchmark_download",
                )
            except Exception as exc:
                st.warning(f"Benchmark CSV could not be read: {exc}")
        else:
            st.info(
                "The historical engine benchmark report was not found at the "
                "expected path. FIRESYNC does not invent or recreate historical results."
            )

        st.caption(
            "Runtime figures are specific to their recorded machine, data and "
            "execution conditions. They are not a general scalability guarantee."
        )

    with methods_tab:
        st.markdown("##### Boundary provenance documentation")

        for document in BOUNDARY_DOCS:
            if document.exists():
                with st.expander(document.name):
                    st.markdown(
                        document.read_text(encoding="utf-8")
                    )
            else:
                st.caption(f"Not found: {document.relative_to(ROOT)}")

        st.divider()
        st.markdown("##### Scientific safeguards")

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

    with schema_tab:
        st.markdown("##### Full normalized dataset schema")

        schema = pd.DataFrame(
            {
                "Column": df_all.columns,
                "Data type": df_all.dtypes.astype(str).values,
                "Non-null values": [
                    int(df_all[column].notna().sum())
                    for column in df_all.columns
                ],
                "Missing values": [
                    int(df_all[column].isna().sum())
                    for column in df_all.columns
                ],
            }
        )

        show_table(schema, 500)

        st.download_button(
            "Download schema · CSV",
            data=schema.to_csv(index=False),
            file_name="firesync_dataset_schema.csv",
            mime="text/csv",
            key="evidence_schema_download",
        )


# ============================================================
# FOOTER
# ============================================================
# ============================================================

st.divider()

st.caption(
    "FIRESYNC | NASA FIRMS historical archive | MODIS + VIIRS | "
    "Research prototype. Satellite thermal anomalies are not "
    "ground-confirmed fire events."
)
