# Reproducing the results

Inputs (not included; download and place in /mnt/user-data/uploads):
- nrsc_flood_extent_2015.parquet, irs_flood_extent_2005.parquet (flood extents)
- FL20151123IND shapefile package (UNOSAT)
- chennai_dem_glo30_32644.tif, chennai_gsw_occurrence_32644.tif (from gee_export_rasters.js)
- chennai_worldcover_v200_32644.tif (from gee_export_worldcover.js)
- gcc_hotspots_2015.csv (parsed from the GCC KML)
- CMA.geojson (DataMeet Municipal_Spatial_Data/Chennai; copy: cma_boundary_datameet.geojson)

Run in this order (scripts are in this folder):
1. raster_compare.py       rasterises NRSC codes and UNOSAT layers on the 30 m grid
2. terrain_step1.py        slope and distance to permanent water from the DEM and occurrence rasters
3. twi90.py                resamples the DEM to 90 m and computes D8 flow accumulation
4. rebuild_code13.py       builds 2015 code-13 positives and background negatives, samples predictors
5. cma_eval.py             clips to the CMA, elevation-matches both events, runs spatial CV and transfer
6. cma_bootstrap.py        bootstrap 95% intervals -> cma_bootstrap_ci.csv
7. results_tables.tex     tables are generated from cma_bootstrap_ci.csv and table_predictor_summary.csv

Notes:
- Random seeds are fixed (sampling seeds 11, 12, 99 in the scripts). Rerunning the matching may change counts slightly if the seed is changed.
- Two inputs were derived in earlier sessions with scripts not included here: build_labels.py (labels.csv), and the Earth Engine export scripts (gee_*.js).
- The 90 m TWI step requires numpy with a compatibility shim for pysheds (np.in1d = np.isin).
