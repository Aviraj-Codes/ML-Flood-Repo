import geopandas as gpd, numpy as np, pandas as pd
from rasterio import features
from rasterio.transform import from_origin
from shapely.validation import make_valid
rng=np.random.default_rng(42)
nrsc=gpd.read_parquet('/mnt/user-data/uploads/nrsc_flood_extent_2015.parquet').to_crs(32644)
irs=gpd.read_parquet('/mnt/user-data/uploads/irs_flood_extent_2005.parquet').to_crs(32644)
irs['geometry']=irs.geometry.apply(make_valid)
minx,miny,maxx,maxy=nrsc.total_bounds
res=30
minx=np.floor(minx/res)*res; miny=np.floor(miny/res)*res
maxx=np.ceil(maxx/res)*res; maxy=np.ceil(maxy/res)*res
W=int((maxx-minx)/res); H=int((maxy-miny)/res)
tf=from_origin(minx,maxy,res,res)
def r(geoms): return features.rasterize([(g,1) for g in geoms], out_shape=(H,W), transform=tf, fill=0, dtype='uint8').astype(bool)
pos2015=r(nrsc[nrsc.pixelvalue==1].geometry)
ex13=r(nrsc[nrsc.pixelvalue==13].geometry)
pos2005=r(irs.geometry)
U24=np.load('flood1124.npy').astype(bool)
pre=np.load('pre_s1.npy').astype(bool)|np.load('pre_ls8.npy').astype(bool)
bg=~(pos2015|ex13|U24|pre)
bg2005=~(pos2005|pos2015)
print('pos2015 km2',round(pos2015.sum()*9e-4,1),' pos2005 km2',round(pos2005.sum()*9e-4,1))
print('background 2015 km2',round(bg.sum()*9e-4,1),' background 2005 km2',round(bg2005.sum()*9e-4,1))
def thin(mask, spacing_cells, n, rng):
    ys,xs=np.nonzero(mask)
    cell=(ys//spacing_cells)*100000+(xs//spacing_cells)
    df=pd.DataFrame({'y':ys,'x':xs,'cell':cell})
    df=df.groupby('cell').sample(n=1, random_state=int(rng.integers(1e9)))
    if len(df)>n: df=df.sample(n=n, random_state=0)
    return df[['y','x']].to_numpy()
rows=[]
def add(idx, flag, year, src):
    ys,xs=idx[:,0],idx[:,1]
    X=minx+(xs+0.5)*res; Y=maxy-(ys+0.5)*res
    rows.append(pd.DataFrame({'x':X,'y':Y,'flood':flag,'event_year':year,'label_source':src}))
p15=thin(pos2015,10,3000,rng)
n15=thin(bg,10,3000,rng)
p05=thin(pos2005,10,1500,rng)
n05=thin(bg2005,10,1500,rng)
add(p15,1,2015,'NRSC_code1'); add(n15,0,2015,'background_no_flood_no_water')
add(p05,1,2005,'IRS_2005'); add(n05,0,2005,'background_outside_IRS_and_2015')
lab=pd.concat(rows,ignore_index=True)
lab.to_csv('labels.csv',index=False)
print(lab.groupby(['event_year','flood']).size().to_string())
print('x range',round(lab.x.min()),round(lab.x.max()),' y range',round(lab.y.min()),round(lab.y.max()))
