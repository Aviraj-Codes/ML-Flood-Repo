import geopandas as gpd, numpy as np, rasterio
from rasterio import features
from rasterio.transform import from_origin
from shapely.geometry import box as B
U='unosat/'
nrsc=gpd.read_parquet('/mnt/user-data/uploads/nrsc_flood_extent_2015.parquet').to_crs(32644)
minx,miny,maxx,maxy=nrsc.total_bounds
res=30
minx=np.floor(minx/res)*res; miny=np.floor(miny/res)*res
maxx=np.ceil(maxx/res)*res; maxy=np.ceil(maxy/res)*res
W=int((maxx-minx)/res); H=int((maxy-miny)/res)
tf=from_origin(minx,maxy,res,res)
print('grid',W,H, 'cells', W*H, flush=True)
def fix(g):
    g=g.copy()
    g['geometry']=g.geometry.buffer(0)
    return g[g.geometry.notnull() & ~g.geometry.is_empty]
def rast(gdf):
    shapes=[(geom,1) for geom in gdf.geometry if geom is not None]
    return features.rasterize(shapes, out_shape=(H,W), transform=tf, fill=0, dtype='uint8') if shapes else np.zeros((H,W),np.uint8)
nr=np.zeros((H,W),np.uint8)
for code in [1,13]:
    sub=nrsc[nrsc.pixelvalue==code]
    m=features.rasterize([(g,1) for g in sub.geometry], out_shape=(H,W), transform=tf, fill=0, dtype='uint8')
    np.save(f'nrsc_{code}.npy', m); print('nrsc',code,'cells',int(m.sum()), flush=True)
bbox=B(minx,miny,maxx,maxy)
bgeo=gpd.GeoSeries([bbox],crs=32644).to_crs(4326).iloc[0].bounds
for name,f in [('flood1124','S1_20151124_Flood.shp'),('flood1112','S1_20151112_Flood.shp'),('pre_s1','S1_20150901_PreFlood.shp'),('pre_ls8','LS8_20151014_PreFlood.shp')]:
    g=gpd.read_file(U+f, bbox=bgeo).to_crs(32644)
    g=fix(g)
    m=rast(g)
    np.save(f'{name}.npy', m); print(name, 'features', len(g), 'cells', int(m.sum()), flush=True)
