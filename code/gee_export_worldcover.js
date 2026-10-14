var region = ee.Geometry.Rectangle([79.30, 12.45, 80.37, 13.40]);

var lulc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').clip(region);

Export.image.toDrive({
  image: lulc,
  description: 'chennai_worldcover_v200_32644',
  folder: 'flood_rasters',
  region: region,
  scale: 30,
  crs: 'EPSG:32644',
  maxPixels: 1e9
});
