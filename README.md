# Flood extent validation and terrain discrimination, Chennai 2015 and 2005

Code and derived results for the paper on built-up concentration of flood extent and within-event terrain discrimination in the Chennai Metropolitan Area, with a failed transfer to the 2005 flood.

## What is included
- `code/`: scripts for label construction, terrain predictors, matched sampling, cross-validation, transfer testing, and bootstrap intervals. Earth Engine exports are in `gee_*.js`.
- `results/`: summary tables used in the paper (bootstrap intervals, single-variable AUCs, predictor summaries).
- `paper/`: LaTeX source and the bibliography (`references_final.bib`).
- `README_run_order.md`: run order and notes.

## What is not included
Input data are not redistributed here. Obtain them from the original sources:
- NRSC and IRS flood extents (Chennai Flood Monitoring and Decision Support System)
- UNOSAT Sentinel-1 flood layers, package FL20151123IND (Humanitarian Data Exchange)
- GCC hotspot and stagnation KML layers (OpenCity)
- Copernicus DEM GLO-30, JRC Global Surface Water, ESA WorldCover v200 (via Google Earth Engine)
- CMA boundary (DataMeet Municipal_Spatial_Data, Chennai/CMA.geojson)

## Reproducing
1. Install dependencies: `pip install -r requirements.txt`
2. Place inputs in the paths named in `README_run_order.md`.
3. Run the scripts in the order given there.

Random seeds are fixed in the scripts. Intervals are 95% bootstrap percentile intervals with 1,000 resamples.

## Known limitations
- Several scripts use hard-coded input paths (`/mnt/user-data/uploads/...`). Change these to your local paths before running.
- The 90 m TWI step requires a numpy compatibility shim for pysheds (`np.in1d = np.isin`).
- The NRSC code definitions and acquisition date are undocumented. The label choice rests on the hotspot comparison.

## Licence
Choose a licence before publishing (for example MIT for the code). Check the terms of each data source before redistributing any derived data.
