var pts = ee.FeatureCollection('users/aviraj2024/labels_lonlat').map(function(f) {
  return f.setGeometry(ee.Geometry.Point([f.getNumber('lon'), f.getNumber('lat')]));
});
print('label count', pts.size());

var dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').mosaic();
var slope = ee.Terrain.slope(dem).rename('slope');

var rain2015 = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterDate('2015-11-01', '2015-12-06').sum().rename('rain_2015');
var rain2005 = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterDate('2005-10-25', '2005-11-20').sum().rename('rain_2005');

var lulc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').rename('lulc');

var clay = ee.Image('projects/soilgrids-isric/clay_mean').select('clay_0-5cm_mean').rename('clay');
var sand = ee.Image('projects/soilgrids-isric/sand_mean').select('sand_0-5cm_mean').rename('sand');
var silt = ee.Image('projects/soilgrids-isric/silt_mean').select('silt_0-5cm_mean').rename('silt');

var water = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').unmask(0).gt(50);
var distWater = water.fastDistanceTransform().sqrt().multiply(ee.Image.pixelArea().sqrt())
  .rename('dist_water_m');

var stack = dem.rename('elevation')
  .addBands([slope, rain2015, rain2005, lulc, clay, sand, silt, distWater]).unmask(-9999);

var sampled = stack.sampleRegions({
  collection: pts,
  properties: ['id', 'flood', 'event_year', 'label_source'],
  scale: 30,
  geometries: true,
  tileScale: 4
});

Export.table.toDrive({
  collection: sampled,
  description: 'flood_predictors_raw',
  fileFormat: 'CSV'
});
