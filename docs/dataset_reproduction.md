
# FIRESYNC — Dataset Reproduction Guide

## 1. Overview

This guide explains how to reproduce the FIRESYNC historical
NASA FIRMS dataset from its four source archive ZIP files.

The pipeline has two stages:

1. Import and validate the original NASA FIRMS archives.
2. Normalize the imported observations into a common schema.

The current reference dataset contains 1,641 observations
covering the requested period 2021–2025.

**Important:** The dataset is an archive sample, not a
demonstration of complete satellite observation coverage.

## 2. Required source archives

Place all four ZIP files in:

`data/archive_downloads/nasa_firms/incoming/`

| Archive filename | NASA FIRMS request ID | Product |
|---|---|---|
| DL_FIRE_M-C61_839652.zip | 839652 | MODIS C6.1 |
| DL_FIRE_J1V-C2_839653.zip | 839653 | VIIRS NOAA-20 C2 |
| DL_FIRE_SV-C2_839654.zip | 839654 | VIIRS Suomi-NPP C2 |
| DL_FIRE_J2V-C2_839655.zip | 839655 | VIIRS NOAA-21 C2 |

Obtain the original archive files through NASA FIRMS.
The archive files are not committed to this repository.

## 3. Reference SHA-256 checksums

Verify the downloaded files against these reference hashes.

| Archive | SHA-256 |
|---|---|
| MODIS C6.1 | `6af3a63a89e9e3fe1941472ba74331eec6295df9d76e1aced4fca84b77ad55ad` |
| NOAA-20 VIIRS C2 | `25713d060203271a43708856aa53f85ff780dd6e4a8b6197f4542d4f66e790a8` |
| Suomi-NPP VIIRS C2 | `75ff33470b16c248d620d4d493adda6715cdd8f678a57c7ad579bf51f9b156f0` |
| NOAA-21 VIIRS C2 | `2a87cebe54b1282c6047a03f5484e012e68ffaa43457c174a102273cc85ab3ce` |

PowerShell verification:

```powershell
Get-ChildItem .\data\archive_downloads\nasa_firms\incoming\*.zip |
    Get-FileHash -Algorithm SHA256 |
    Format-List Path,Hash
```

The calculated hashes must match the reference values.
Do not use an archive whose checksum does not match.

## 4. Environment setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Import and validate archives

Run:

```powershell
python scripts/import_firms_archives.py
```

The importer checks:

- Archive ZIP integrity.
- Exactly one CSV member per archive.
- Required NASA FIRMS columns.
- Source archive checksums.
- Coordinate validity.
- Observation metadata and provenance.

Expected outputs:

- `data/interim/firms_archive_raw_2021_2025.parquet`
- `data/interim/firms_archive_import_report.json`

## 6. Normalize observations

After successful import, run:

```powershell
python scripts/normalize_firms_archives.py
```

The normalization stage preserves sensor-specific attributes,
platform identities, native confidence schemes and observation
provenance.

Expected outputs:

- `data/processed/firms_archive_normalized_2021_2025.parquet`
- `data/interim/firms_archive_normalization_report.json`

## 7. Reference dataset validation

The currently verified reference outputs contain:

- Total observations: 1,641
- Unique observation IDs: 1,641
- MODIS observations: 80
- VIIRS observations: 1,561
- Invalid UTC timestamps: 0
- Invalid coordinates in archive import: 0

Platform counts:

| Platform | Observations |
|---|---:|
| Aqua | 62 |
| Terra | 18 |
| NOAA-20 | 602 |
| NOAA-21 | 345 |
| Suomi-NPP | 614 |

These are reference results from the currently validated
archive files. Different source archives may produce different
results.

## 8. Run the application

Once the normalized Parquet file exists:

```powershell
streamlit run app.py
```

## 9. Existing-output protection

Both processing scripts refuse to overwrite existing outputs.

If an output already exists, the script raises a
`FileExistsError`.

For a fresh reproduction, use a clean working copy or
carefully back up and remove only the generated outputs
before rerunning the pipeline.

Do not delete source archives or unrelated research data.

## 10. Scientific limitations

- The dataset is an imported archive sample, not a proven
  complete record of all satellite observations.
- Observation counts vary substantially across dates,
  months and satellite platforms.
- A month with zero retrieved observations does not prove
  that no fire occurred.
- MODIS and VIIRS confidence schemes are not interchangeable.
- A satellite hotspot is a thermal observation, not
  automatically a confirmed wildfire.
- Archive validation does not independently establish
  processing-quality classification.
- The Bandipur landscape acquisition rectangle is not an
  official protected-area boundary.

FIRESYNC is an exploratory scientific investigation platform,
not an operational wildfire alert or emergency-response system.

## 11. Data and credentials policy

The `data/` directory is excluded from Git.

Do not commit:

- NASA FIRMS API keys.
- `.env` files.
- Downloaded archives.
- Generated Parquet datasets.
- Other private credentials.
