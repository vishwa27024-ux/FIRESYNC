# FIRESYNC

## Sensor-Aware Satellite Fire Intelligence System

**Two satellites. Different views. One clearer picture of fire activity.**

FIRESYNC is a scientific investigation platform that explores how MODIS and VIIRS satellite active-fire observations differ, overlap, and can be compared responsibly.

It transforms historical satellite observations into interactive maps, activity calendars, candidate associations, and transparent evidence.

> A satellite hotspot is a detected thermal anomaly, not automatically a confirmed wildfire or a unique fire event.

## Mission

Build a scientifically transparent and reproducible application for understanding historical active-fire observations while accounting for differences in satellite sensors, spatial resolution, acquisition timing, and confidence schemes.

## Study Region

**Initial study region:** Bandipur landscape, Karnataka, India.

The working acquisition rectangle is used for regional analysis. It must not be interpreted as an official protected-area boundary. Available boundary geometries have documented differences and remain provisional.

## Implemented Features

FIRESYNC currently provides seven application modules:

1. **Overview** — Dataset summary, sensor distribution, and platform information.
2. **Fire Map Explorer** — Interactive geographical exploration of satellite observations.
3. **Activity Calendar** — Daily and monthly observation patterns.
4. **Association Explorer** — Spatially and temporally constrained MODIS–VIIRS candidate associations.
5. **Harmonization Lab** — Sensor characteristics, confidence schemes, sensitivity analysis, and engine benchmarking.
6. **Activity Investigation** — Individual observation and candidate-cluster investigation.
7. **Evidence Centre** — Dataset metadata, benchmark reports, boundary documentation, and scientific safeguards.

## Dataset

The current local historical dataset contains:

- **1,641 normalized observations**
- **Observation period:** 2021–2025
- **MODIS:** 80 observations
- **VIIRS:** 1,561 observations
- **Platforms:** Aqua, Terra, Suomi-NPP, NOAA-20, and NOAA-21

The dataset was imported from NASA FIRMS archive products and normalized while preserving sensor and platform identities.

**Coverage limitation:** This is an imported archive sample, not a demonstrated complete record of all satellite observations or fire activity across the study period. Observation counts vary substantially by month and year. A month with zero retrieved observations does not establish that no fire occurred or that satellite coverage was available.

The processed dataset is stored locally at:

`data/processed/firms_archive_normalized_2021_2025.parquet`

The `data/` directory is excluded from Git. The dataset must therefore be supplied or generated separately before running the application.

## Scientific Methodology and Safeguards

- Preserve original observation identities and native sensor attributes.
- Keep MODIS and VIIRS confidence schemes separate.
- Do not treat nominal sensor resolutions as interchangeable.
- Use UTC-aware acquisition timestamps.
- Apply explicit spatial-distance and temporal-window thresholds when generating candidate associations.
- Use geodesic distance checks for candidate association.
- Treat connected components as unconfirmed candidate clusters, not confirmed fire events.
- Do not infer satellite pixel geometry from scan and track values.
- Do not treat H3/grid aggregation as equivalent to native sensor footprint geometry.
- Do not interpret zero retrieved observations as proof of no fire or no satellite coverage.
- Document uncertainty, assumptions, data limitations, and boundary provenance.

## Association Engine Validation

FIRESYNC includes two independently implemented association engines:

- Reference association engine.
- Spatially optimized association engine using a spatial index followed by exact distance and temporal checks.

On the current 1,641-observation dataset, the historical comparison tested 16 parameter configurations with three repetitions. Both engines produced identical candidate edges, cluster memberships, and unmatched observations for the tested configurations.

The historical benchmark measured a median speedup of approximately **4.41×** for the optimized engine on this dataset. Runtime results are environment- and dataset-dependent.

At the default investigation thresholds of 1 km and 60 minutes, the current dataset produces:

- 131 candidate edges
- 30 connected components
- 137 observations in candidate clusters
- 1,504 unmatched observations

These are candidate associations only. They do not establish that observations represent the same physical fire.

## Technology

- Python 3.12
- Streamlit
- Pandas and NumPy
- GeoPandas and Shapely
- DuckDB and Parquet
- SciPy
- Plotly and PyDeck
- Pytest

## Local Setup

### 1. Clone the repository

```powershell
git clone https://github.com/vishwa27024-ux/FIRESYNC.git
cd FIRESYNC
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare the dataset

Obtain or generate the normalized historical dataset and place it at:

```text
data/processed/firms_archive_normalized_2021_2025.parquet
```

NASA FIRMS access credentials, if required by the acquisition workflow, must be configured locally. Never commit API keys, `.env` files, or other secrets.

### 5. Run FIRESYNC

```powershell
streamlit run app.py
```

## Testing

Run the automated test suite:

```powershell
python -m pytest -q
```

Check Python source compilation:

```powershell
python -m compileall -q app.py src scripts
```

Latest verified regression result:

- 25 tests passed.
- Compilation check completed without errors.

## Project Status

The core application, historical dataset normalization, candidate association engines, scientific audits, and seven application modules are implemented.

The project is undergoing final documentation, reproducibility, presentation, and demonstration-readiness work.

## Responsible Interpretation

FIRESYNC is an exploratory scientific analysis and visualization tool. It is not an operational wildfire alert service, emergency-response system, fire-spread predictor, or substitute for authoritative fire monitoring.

All interpretations must account for sensor differences, acquisition timing, confidence limitations, incomplete data coverage, and uncertainty.

---

**FIRESYNC — Sensor-aware satellite fire intelligence.**
