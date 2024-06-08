# MGWR Analysis and Visualization in Google Earth Engine

This repository contains a Python package for conducting Multi-Scale Geographically Weighted Regression (MGWR) analysis on Google Earth Engine (GEE) images and visualizing the results.

## Package

The `gee_mgwr` package includes the following functions:

| Function | Description |
| --- | --- |
| `gee_to_mgwr(gee_image, dependent_variable, independent_variables, scale, region_of_interest, projection)` | Performs MGWR analysis on a GEE image. |
| `package_box_region(gee_image, scale, region_of_interest, mgwr_results, dependent_variable_data, independent_variables_data, projection)` | Packages the results of an MGWR analysis into a format suitable for image conversion. |
| `mgwr_to_ee(image, DV_band, IV_bands, scales, roi, projection)` | Converts MGWR results to Earth Engine images for multiple scales. |
| `get_min_max_vis(image, band_name, scale=None, region=None)` | Utility function for visualizing the MGWR results in Google Earth Engine. |
| `mgwr_add_to_map(collection, band_name, region, project=False, projection=None, num_images=1)` | Adds MGWR results to a map for visualization. |


## Dependencies

- `numpy`
- `pandas`
- `geemap`
- `ee` (Earth Engine API)
- `matplotlib`
- `mgwr`

## Installation

You can install the `gee_mgwr` package directly from GitHub using pip:

```bash
!pip install git+https://github.com/neillinehan/gee_mgwr.git
```
## Usage

1. Ensure you have the required dependencies installed.
2. Import the `gee_mgwr` package:

```bash
import gee_mgwr
```

## License

This project is licensed under the MIT License.

