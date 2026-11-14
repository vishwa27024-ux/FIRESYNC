# FIRESYNC Harmonization & Calibration Methodology

## Scientific Rationale (PRD FR-05)

Comparing raw MODIS and VIIRS active-fire detection counts without calibration introduces severe observational bias due to fundamental sensor differences:

1. **Spatial Footprint Disparity**:
   - MODIS: Nominal 1 km resolution at nadir ($\sim 1.0\text{ km}^2$ per pixel).
   - VIIRS: Nominal 375 m resolution at nadir ($\sim 0.14\text{ km}^2$ per pixel).
   - The pixel area ratio at nadir is approximately $7.1\times$, which expands even further toward the swath edges. Consequently, a single physical fire perimeter often triggers multiple adjacent VIIRS detections while registering as only one MODIS pixel.

2. **Sub-Pixel Radiative Power Detection Threshold**:
   - Under typical atmospheric conditions, MODIS requires a fire radiative power of approximately $\text{FRP} \ge 10\text{ MW}$ to reliably trigger an active-fire anomaly.
   - VIIRS 375 m I-band detections routinely identify smoldering or small fires with $\text{FRP} \ge 0.5\text{–}1.0\text{ MW}$.
   - On low-intensity fire days, VIIRS frequently records fire activity where MODIS registers zero detections.

3. **Risk of Misleading Trends**:
   - A naive historical comparison between the pre-VIIRS era (2000–2011) and the VIIRS era (2012–present) creates the false impression of an explosion in wildfire activity.

---

## Analytical Formulation

### Analytical Units

To compare observations without making unsubstantiated assumptions about pixel boundaries, FIRESYNC aggregates observations into space-time analytical units:
- **Daily Landscape Analytical Units**: Daily temporal bins across the Bandipur study landscape, capturing total counts, FRP distributions (sum, mean, max), and overpass day/night proportions.
- **Spatio-Temporal Grid Units**: Configurable $0.05^\circ \times 0.05^\circ$ ($\sim 5.5\text{ km}$) analytical grid cells.

---

## Calibration Model Hierarchy

### 1. Empirical Area & Ratio Scaling (Baseline)
Uses the concurrent observation ratio $\bar{R}$:
$$\bar{R} = \frac{\sum_{\text{concurrent}} N_{\text{VIIRS}}}{\sum_{\text{concurrent}} N_{\text{MODIS}}}$$
$$\hat{N}_{\text{MODIS}} = \frac{N_{\text{VIIRS}}}{\max(\bar{R}, 10^{-6})}$$

### 2. Robust Log-Log Huber Regression (Baseline Statistical)
Power-law relationship between VIIRS count, cumulative FRP, and peak FRP, fitted using Huber loss ($\epsilon = 1.35$) to prevent severe leverage from extreme outlier fire events:
$$\ln(\hat{N}_{\text{MODIS}} + 1) = \beta_0 + \beta_1 \ln(N_{\text{VIIRS}} + 1) + \beta_2 \ln(\text{FRP}_{\text{sum}} + 1) + \beta_3 \ln(\text{FRP}_{\text{max}} + 1)$$
$$\hat{N}_{\text{MODIS}} = \max\left(0, \exp(\hat{y}_{\text{log}}) - 1\right)$$

### 3. Two-Stage Hurdle Model
Accounts for the zero-inflated nature of MODIS observations:
1. **Occurrence Stage**: Logistic regression estimates the probability that a fire exceeds the MODIS detection threshold:
   $$P(N_{\text{MODIS}} > 0 \mid \mathbf{x}_{\text{VIIRS}}) = \sigma(\mathbf{w}^T \mathbf{x})$$
2. **Intensity Stage**: Ridge regression estimates expected detections on active days:
   $$\mathbb{E}[N_{\text{MODIS}} \mid N_{\text{MODIS}} > 0] = \exp(\boldsymbol{\theta}^T \mathbf{x})$$
3. **Harmonized Estimate**:
   $$\hat{N}_{\text{MODIS}} = P(N_{\text{MODIS}} > 0) \times \mathbb{E}[N_{\text{MODIS}} \mid N_{\text{MODIS}} > 0]$$

### 4. Non-Linear Random Forest Regressor
Multi-feature ensemble model combining count, FRP sum, FRP mean, FRP max, day/night ratio, and seasonality month:
$$\hat{N}_{\text{MODIS}} = \text{RandomForest}(\mathbf{x}_{\text{VIIRS}})$$

---

## Uncertainty Quantification

Every calibrated prediction $\hat{N}_i$ is bounded by a 95% Prediction Interval:
$$[\text{CI}_{\text{lower}}^{95}, \text{CI}_{\text{upper}}^{95}] = \left[\max(0, \hat{N}_i - 1.96 \cdot \hat{\sigma}_{\text{res}}), \; \hat{N}_i + 1.96 \cdot \hat{\sigma}_{\text{res}}\right]$$
where $\hat{\sigma}_{\text{res}}$ is the out-of-fold residual standard deviation.

---

## Independent Validation Scheme

- **Chronological Split**: Observations prior to 2024 form the training/calibration set; observations from 2024–2025 form the independent holdout validation set.
- **Evaluation Metrics**:
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - Coefficient of Determination ($R^2$)
  - Mean Residual Bias ($\bar{\epsilon} = \frac{1}{n} \sum (\hat{y}_i - y_i)$)

---

## Scientific Safeguards

1. Calibrated values never replace or overwrite raw NASA FIRMS records.
2. Every calibrated record carries a `data_quality_flag`:
   - `CONCURRENT_CALIBRATED`: Concurrent MODIS & VIIRS detections verified on the same date.
   - `VIIRS_IMPUTED_MODIS_EQUIVALENT`: Estimated equivalent activity when only VIIRS observed.
   - `MODIS_ONLY_OBSERVED`: Native MODIS observation where VIIRS was not present.
   - `NO_DETECTION`: Zero active fires detected.
3. Satellite thermal anomalies remain unconfirmed candidate fire detections, not confirmed ground wildfires.
