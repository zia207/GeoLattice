# colibrì: qwen3.6-colibri

11/20/2026, 8:05:29 PM

## You

How to Estimate ET from Landsat, SEntinel and MODIS using Python

## colibrì

Estimating Evapotranspiration (ET) from satellite data is a complex process that typically relies on the **Surface Energy Balance (SEB)** approach. The most common method uses the **SEBAL** (Surface Energy Balance Algorithm for Land) or **METRIC** (Mapping Evapotranspiration at High Resolution with Internal Calibration) algorithms.

Below is a structured guide on how to estimate ET from **Landsat**, **Sentinel-2**, and **MODIS** using Python, including the necessary steps, libraries, and code examples.

---

## 🌍 Key Concepts

1. **Evapotranspiration (ET)**: The sum of evaporation from the land surface plus transpiration from plants.
2. **Remote Sensing ET Methods**:
   - **SEBAL/METRIC**: Uses surface energy balance equations.
   - **Priestley-Taylor**: Uses radiation and temperature.
   - **Penman-Monteith**: Requires meteorological data + remote sensing inputs.
3. **Satellite Data Characteristics**:
   - **Landsat (8/9)**: 30m resolution, 16-day revisit. Best for detailed ET.
   - **Sentinel-2 (1/2)**: 10–20m resolution, 5-day revisit. High spatial detail.
   - **MODIS**: 500m–1km resolution, daily revisit. Good for temporal trends.

---

## 🛠️ Python Libraries Needed

```bash
pip install rasterio numpy scipy xarray geemap earthengine-api matplotlib seaborn
```

- **`rasterio`**: Read/write raster files.
- **`numpy`**: Array operations.
- **`geemap`**: Google Earth Engine interface (optional but powerful).
- **`xarray`**: Handle multi-dimensional arrays.
- **`scipy`**: Scientific computing (interpolation, optimization).

---

## 📡 Step-by-Step Workflow

### 1. Data Acquisition

#### Option A: Using Google Earth Engine (GEE) via `geemap`
```python
import geemap

# Create a map
m = geemap.Map(center=[20, 0], zoom=2)

# Load Landsat 8 Surface Reflectance
landsat = geemap.ee_load_image(
    'LANDSAT/LC08/C02/T1_L2',
    start_date='2023-01-01',
    end_date='2023-01-15',
    region=[-120, 30, -110, 40],  # Example: California
    bands=['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
)

m.add_layer(landsat, {'bands': ['SR_B5', 'SR_B4', 'SR_B3'], 'min': 0, 'max': 0.3}, 'Landsat 8')
m
```

#### Option B: Downloading from USGS EarthExplorer or ESA Copernicus
Use `earthengine-api` or `requests` to download TIFF files, then read with `rasterio`.

```python
import rasterio

with rasterio.open('LC08_L2SR_032037_20230101_20230101_02_T1_B4.TIF') as src:
    band4 = src.read(1)  # Near-Infrared
```

---

### 2. Preprocessing

#### A. Cloud Masking
Use the `CFMask` or `QA` band to mask clouds.

```python
def mask_clouds(landsat_sr, qa_band):
    # Bit 5 is cloud
    cloud_bit_mask = 1 << 5
    # Bit 6 is cirrus
    cirrus_bit_mask = 1 << 3
    mask = (qa_band & cloud_bit_mask == 0) & (qa_band & cirrus_bit_mask == 0)
    return landsat_sr * mask
```

#### B. Atmospheric Correction
If using Top-of-Atmosphere (TOA) reflectance, convert to Surface Reflectance using `py6S` or `ATCor`.

```python
# Example using py6S
from py6S import SixS
s = SixS()
s.atmos_profile = SixS.MidLatitudeSummer
s.aerosol_profile = SixS.Maritime
s.geometry = SixS.Satellite
s.geometry.month = 7
s.geometry.day = 15
s.geometry.gmt_time = 12
s.geometry.latitude = 35
s.geometry.longitude = -120
s.run()
# s.toa_reflectance will give corrected values
```

#### C. Resampling
Ensure all bands are in the same spatial resolution and projection.

```python
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

# Example: Resample MODIS (500m) to Landsat (30m) resolution if needed
```

---

### 3. Calculate ET Using SEBAL-like Approach

The SEBAL equation for ET is:

$$
\lambda ET = R_n - G - H
$$

Where:
- $\lambda ET$: Latent heat flux (W/m²)
- $R_n$: Net radiation
- $G$: Soil heat flux
- $H$: Sensible heat flux

#### Step 3.1: Calculate Land Surface Temperature (LST)
From thermal bands (Landsat Band 10/11, Sentinel-2 no thermal, MODIS Bands 31/32).

```python
import numpy as np

def calculate_lst(b10, b11, k1, k2, emissivity=0.98):
    # Convert DN to TOA Radiance
    # L = M_L * Q_cal + A_L
    # L = (M_B10 * Q_cal + A_B10)
    # Convert Radiance to Brightness Temperature
    # T = K2 / ln(K1/L + 1)
    # Convert to Kelvin, then Celsius
    pass
```

#### Step 3.2: Calculate Albedo
```python
def calculate_albedo(b2, b3, b4, b5, b6, b7):
    # Simplified albedo from Landsat 8 bands
    albedo = 0.29 * b2 + 0.08 * b3 + 0.04 * b4 + 0.07 * b5 + 0.12 * b6 + 0.32 * b7
    return albedo
```

#### Step 3.3: Calculate Net Radiation ($R_n$)
```python
def calculate_net_radiation(shortwave_radiation, albedo, lst, emissivity=0.98):
    # R_n = (1 - albedo) * R_s + L_down - L_up
    # L_up = emissivity * sigma * T^4
    sigma = 5.67e-8  # Stefan-Boltzmann constant
    lst_kelvin = lst + 273.15
    l_up = emissivity * sigma * (lst_kelvin ** 4)
    # Assume L_down ≈ shortwave_radiation * 0.5 (simplified)
    l_down = shortwave_radiation * 0.5
    rn = (1 - albedo) * shortwave_radiation + l_down - l_up
    return rn
```

#### Step 3.4: Estimate Sensible Heat Flux ($H$)
This requires empirical relationships between LST and temperature gradients.

```python
def calculate_sensible_heat_flux(rn, g, t_s, t_a, wind_speed):
    # H = 0.0025 * (t_s - t_a)^2 * wind_speed (simplified)
    h = 0.0025 * (t_s - t_a)**2 * wind_speed
    return h
```

#### Step 3.5: Calculate ET
```python
def calculate_et(rn, g, h):
    # lambda ET = R_n - G - H
    # Convert W/m² to mm/day: 1 W/m² = 86.4 mm/day
    et_mm_day = (rn - g - h) * 86.4 / 2260  # 2260 kJ/kg is latent heat of vaporization
    return et_mm_day
```

---

## 📊 Specifics for Each Satellite

### Landsat 8/9
- **Advantages**: High spatial resolution (30m), thermal bands (10/11).
- **Challenges**: 16-day revisit, cloud cover.
- **Use Case**: Detailed agricultural ET.

### Sentinel-2
- **Advantages**: High spatial resolution (10–20m), frequent revisit (5 days).
- **Challenges**: No thermal bands. Must use LST from Landsat or MODIS, or use machine learning to estimate LST from visible/NIR bands.
- **Workaround**: Use **LST from Landsat** and **vegetation indices from Sentinel-2** in a hybrid model.

### MODIS
- **Advantages**: Daily coverage, global.
- **Challenges**: Low spatial resolution (500m–1km), less accurate for small fields.
- **Use Case**: Regional ET monitoring, drought assessment.

---

## 🧪 Example: Full Pipeline with `geemap` and `rasterio`

```python
import geemap
import numpy as np
import rasterio
from rasterio.windows import from_bounds

# 1. Load data via GEE
m = geemap.Map()
landsat = geemap.ee_load_image(
    'LANDSAT/LC08/C02/T1_L2',
    start_date='2023-06-01',
    end_date='2023-06-15',
    region=[-120, 30, -110, 40],
    bands=['SR_B4', 'SR_B5', 'SR_B6', 'ST_B10']
)

# 2. Export to local drive (optional)
# geemap.ee_export_image(landsat, filename='landsat.tif', scale=30)

# 3. Read with rasterio
with rasterio.open('landsat.tif') as src:
    nir = src.read(1)
    red = src.read(2)
    swir = src.read(3)
    ltb = src.read(4)  # Thermal band

# 4. Calculate NDVI
ndvi = (nir - red) / (nir + red)

# 5. Calculate LST from thermal band (simplified)
# L = 0.00388324 * ltb - 151.2405
# T = L / (0.00388324 * 1000000)  # Simplified
lst = (ltb * 0.00341802) + 149.0  # Simplified formula

# 6. Calculate ET using a simplified Priestley-Taylor approach
# ET = alpha * (Rn - G) / (lambda * rho * cp) * (delta / (delta + gamma))
# For simplicity, assume alpha = 1.26
# This is a very simplified example.

# 7. Visualize
m.add_layer(ndvi, {'bands': ['NDVI'], 'min': 0, 'max': 1}, 'NDVI')
m
```

---

## 📚 Recommended Tools & Packages

1. **`pySEBAL`**: Python implementation of SEBAL.
   ```bash
   pip install pysebal
   ```
2. **`ETScan`**: Open-source ET tool.
3. **`Google Earth Engine`**: For large-scale processing.
4. **`Sentinel Hub`**: For Sentinel-2 data processing.

---

## ⚠️ Challenges & Considerations

1. **Cloud Cover**: Landsat and Sentinel-2 are affected by clouds. Use cloud masking and compositing.
2. **Thermal Data**: Sentinel-2 lacks thermal bands. Use Landsat LST or MODIS LST.
3. **Meteorological Data**: ET models require air temperature, humidity, wind speed. Use ERA5 or station data.
4. **Validation**: Ground truth data is essential for calibration.
5. **Computational Cost**: High-resolution ET calculations are computationally intensive. Use parallel processing or GEE.

---

## ✅ Summary

| Satellite | Resolution | Revisit | Thermal? | Best Use Case |
|-----------|------------|---------|----------|---------------|
| Landsat 8/9 | 30m | 16 days | Yes | Detailed agricultural ET |
| Sentinel-2 | 10–20m | 5 days | No | High-res vegetation monitoring |
| MODIS | 500m–1km | Daily | Yes | Regional/drought monitoring |

To estimate ET accurately:
1. Use **Landsat** for high-resolution, single-date ET.
2. Use **Sentinel-2** for frequent monitoring (combine with Landsat LST).
3. Use **MODIS** for daily temporal trends.

For production-grade ET, consider using **Google Earth Engine** with pre-built ET algorithms (e.g., `MOD16A2H`, `ETLook`) or specialized Python libraries like `pySEBAL`.

## You

Write an outline of a Python Tutorial on Estimating Evapotranspiration (ET) from satellite data  - Surface Energy Balance Algorithm for Land, mapping Evapotranspiration at high Resolution with an internalized calibration model - https://etdata.org/methods/

## colibrì

Here is a comprehensive outline for a Python tutorial focused on estimating Evapotranspiration (ET) using the **SEBAL** (Surface Energy Balance Algorithm for Land) and **METRIC** (Mapping Evapotranspiration at High Resolution with an Internal Calibration) models.

This tutorial is designed for intermediate Python users with basic knowledge of remote sensing and environmental science.

---

# 📘 Tutorial Title: **Estimating Evapotranspiration with Python: SEBAL & METRIC Models**

## 🎯 **Learning Objectives**
By the end of this tutorial, participants will be able to:
1. Understand the theoretical foundations of Surface Energy Balance (SEB) models.
2. Differentiate between SEBAL and METRIC methodologies.
3. Process Landsat/Sentinel data for ET estimation using Python.
4. Implement internal calibration techniques (METRIC) to reduce meteorological data dependency.
5. Visualize and validate ET results.

---

## 📅 **Module 1: Introduction to Remote Sensing ET**

### 1.1 What is Evapotranspiration (ET)?
- Definition: Evaporation + Transpiration.
- Importance: Water resource management, agriculture, climate modeling.

### 1.2 Remote Sensing Approaches for ET
- **Empirical Models**: Priestley-Taylor, Hargreaves.
- **Physical Models**: Surface Energy Balance (SEB) – SEBAL, METRIC, SEBS.
- **Hybrid Models**: Combining RS with meteorological data.

### 1.3 Why SEBAL and METRIC?
- **SEBAL**: Robust, requires meteorological data.
- **METRIC**: Internally calibrated, reduces need for ground weather data, uses "hot/cold" pixel anchors.

---

## 🛠️ **Module 2: Python Environment & Libraries Setup**

### 2.1 Recommended Environment
- Conda environment setup.
- Required libraries:
  ```bash
  pip install rasterio numpy scipy xarray geemap earthengine-api matplotlib seaborn pysebal
  ```

### 2.2 Data Sources
- **Landsat 8/9**: USGS Earth Explorer, GEE.
- **Sentinel-2**: Copernicus Open Access Hub, GEE.
- **Meteorological Data**: ERA5, NOAA, or local weather stations.

---

## 📡 **Module 3: Data Acquisition & Preprocessing**

### 3.1 Loading Satellite Data
- Using `geemap` to load Landsat/Sentinel data from GEE.
- Using `rasterio` to load local TIFF files.

### 3.2 Cloud Masking
- Implementing CFMask or QA band-based cloud masking.
- Code Example:
  ```python
  def mask_clouds(landsat_sr, qa_band):
      cloud_bit = 1 << 5
      mask = (qa_band & cloud_bit == 0)
      return landsat_sr * mask
  ```

### 3.3 Atmospheric Correction
- Converting TOA Reflectance to Surface Reflectance.
- Using `py6S` or `ATCor` for accurate correction.

### 3.4 Resampling & Reprojection
- Aligning bands to a common grid (e.g., 30m for Landsat).
- Handling missing values and nodata.

---

## 🌡️ **Module 4: Surface Energy Balance Fundamentals**

### 4.1 The SEB Equation
$$
R_n = H + LE + G
$$
Where:
- $R_n$: Net Radiation
- $H$: Sensible Heat Flux
- $LE$: Latent Heat Flux (ET)
- $G$: Soil Heat Flux

### 4.2 Key Variables to Estimate
1. **Net Radiation ($R_n$)**:
   - Shortwave radiation balance.
   - Longwave radiation balance.
2. **Soil Heat Flux ($G$)**:
   - Empirical relationship with $R_n$ and NDVI.
3. **Sensible Heat Flux ($H$)**:
   - Depends on temperature gradient and wind speed.

---

## 🔬 **Module 5: Implementing SEBAL in Python**

### 5.1 Step 1: Calculate Land Surface Temperature (LST)
- Using thermal bands (Landsat B10/B11).
- Code Example:
  ```python
  def calculate_lst(b10, k1, k2, emissivity=0.98):
      radiance = (b10 - k1) / k2
      brightness_temp = k1 / np.log(radiance + 1)
      lst = brightness_temp - 273.15  # Convert to Celsius
      return lst
  ```

### 5.2 Step 2: Calculate Albedo & NDVI
- Albedo from spectral bands.
- NDVI for vegetation index.

### 5.3 Step 3: Estimate Net Radiation ($R_n$)
- Using shortwave radiation and albedo.

### 5.4 Step 4: Estimate Soil Heat Flux ($G$)
- Using empirical formulas based on NDVI and $R_n$.

### 5.5 Step 5: Estimate Sensible Heat Flux ($H$)$
- Requires air temperature ($T_a$) and wind speed.
- Using temperature gradient between surface and air.

### 5.6 Step 6: Calculate ET
- Solving for $LE = R_n - G - H$.
- Converting W/m² to mm/day.

---

## 🎯 **Module 6: Implementing METRIC (Internal Calibration)**

### 6.1 Why METRIC?
- Reduces dependency on meteorological data.
- Uses "hot" and "cold" pixel anchors for calibration.

### 6.2 Step 1: Identify Hot and Cold Pixels
- **Cold Pixel**: Fully irrigated/vegetated area (maximum ET).
- **Hot Pixel**: Bare soil/dry area (minimum ET, maximum $H$).

### 6.3 Step 2: Calculate Anchor Points
- Compute $H$ and $LE$ for hot and cold pixels using SEBAL-like equations.

### 6.4 Step 3: Linear Interpolation
- Map $H$ across the image using linear interpolation between hot and cold pixels.
- Code Example:
  ```python
  # Normalize LST to [0, 1]
  lst_norm = (lst - lst_min) / (lst_max - lst_min)
  # Interpolate H
  H = H_hot + (H_cold - H_hot) * lst_norm
  ```

### 6.5 Step 4: Calculate ET
- $LE = R_n - G - H$.

---

## 📊 **Module 7: Visualization & Validation**

### 7.1 Visualizing ET Results
- Using `matplotlib` and `seaborn` for maps.
- Code Example:
  ```python
  import matplotlib.pyplot as plt
  plt.imshow(et_mm_day, cmap='YlGn')
  plt.colorbar(label='ET (mm/day)')
  plt.title('Estimated Evapotranspiration')
  plt.show()
  ```

### 7.2 Validation with Ground Data
- Comparing RS-ET with flux tower data.
- Metrics: RMSE, R², Bias.

### 7.3 Temporal Analysis
- Creating ET time series for drought monitoring.

---

## 🚀 **Module 8: Advanced Topics & Best Practices**

### 8.1 Handling Missing Data
- Interpolation techniques.
- Using MODIS for daily ET estimates.

### 8.2 Scaling to Regional Levels
- Using GEE for large-area processing.
- Parallel processing with `dask`.

### 8.3 Uncertainty Analysis
- Sensitivity to input parameters (albedo, emissivity, etc.).

### 8.4 Integration with ETData.org
- Comparing results with ETData.org methods.
- Understanding differences in model assumptions.

---

## 📚 **Appendix**

### A.1 Code Repository
- GitHub link with complete notebooks.

### A.2 Glossary
- Definitions of key terms (e.g., Albedo, LST, SEB).

### A.3 References
- Allen et al. (2007) – SEBAL Manual.
- Allen et al. (1998) – METRIC Model.
- ETData.org Methodologies.

---

## ✅ **Final Project**
- **Task**: Estimate ET for a specific region using Landsat data.
- **Steps**:
  1. Load and preprocess data.
  2. Implement SEBAL or METRIC.
  3. Visualize results.
  4. Validate with available ground data.

---

This outline provides a structured path from theory to implementation, ensuring participants gain practical skills in estimating ET using advanced Python techniques.
