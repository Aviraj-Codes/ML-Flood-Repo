var region = ee.Geometry.Rectangle([79.30, 12.45, 80.37, 13.40]);

var dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').mosaic().clip(region);

var occ = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').unmask(0).clip(region);

Export.image.toDrive({
  image: dem,
  description: 'chennai_dem_glo30_32644',
  folder: 'flood_rasters',
  region: region,
  scale: 30,
  crs: 'EPSG:32644',
  maxPixels: 1e9
});

Export.image.toDrive({
  image: occ,
  description: 'chennai_gsw_occurrence_32644',
  folder: 'flood_rasters',
  region: region,
  scale: 30,
  crs: 'EPSG:32644',
  maxPixels: 1e9
});
