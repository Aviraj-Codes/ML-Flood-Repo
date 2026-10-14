import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from rasterio.transform import from_origin, rowcol
rng=np.random.default_rng(21)
elev=np.load('elev_nrsc_grid.npy')
import geopandas as gpd
from scipy import ndimage
nrsc=gpd.read_parquet('/mnt/user-data/uploads/nrsc_flood_extent_2015.parquet').to_crs(32644)
minx,miny,maxx,maxy=nrsc.total_bounds
res=30; minx=np.floor(minx/res)*res; maxy=np.ceil(maxy/res)*res
n1=np.load('nrsc_1.npy').astype(bool); n13=np.load('nrsc_13.npy').astype(bool)
U=np.load('flood1124.npy').astype(bool)
pre=np.load('pre_s1.npy').astype(bool)|np.load('pre_ls8.npy').astype(bool)
bg=~(ndimage.binary_dilation(n13,iterations=10)|ndimage.binary_dilation(n1,iterations=10)|U|pre)
cand=bg&np.isfinite(elev)&(elev>0)
pos=pd.read_csv('code13_2015_points.csv')
bins=np.arange(0,31,1)
pc,_=np.histogram(pos[pos.flood==1].elevation.clip(0,30),bins=bins)
ii,jj=np.nonzero(cand)
ce=elev[ii,jj].clip(0,30)
bin_idx=np.digitize(ce,bins)-1
pick=[]
for b in range(len(bins)-1):
    need=pc[b]
    if need==0: continue
    idx=np.nonzero(bin_idx==b)[0]
    sel=rng.choice(idx,size=need,replace=False)
    pick.append(sel)
pick=np.concatenate(pick)
rows=ii[pick]; cols=jj[pick]
x=minx+(cols+0.5)*res; y=maxy-(rows+0.5)*res
neg=pd.DataFrame({'x':x,'y':y,'flood':0,'event_year':2015,'elevation':elev[rows,cols]})
pos_only=pos[pos.flood==1][['x','y','flood','event_year','elevation','slope','twi','dist_water_m','gsw','block']].copy() if 'gsw' in pos.columns else pos[pos.flood==1].copy()
dem=np.load('dem.npy'); slope=np.load('slope_deg.npy'); dist=np.load('dist_water.npy'); occ=np.load('occ.npy')
r,c=rowcol(from_origin(315210,1482030,30,30),neg.x.values,neg.y.values); r=np.array(r); c=np.array(c)
neg['slope']=slope[r,c]; neg['dist_water_m']=dist[r,c]; neg['gsw']=occ[r,c]
twi=np.load('twi90.npy'); r9,c9=rowcol(from_origin(315210,1482030,90,90),neg.x.values,neg.y.values)
r9=np.array(r9); c9=np.array(c9); neg['twi']=twi[r9,c9]
neg=neg.dropna(subset=['slope','twi','dist_water_m'])
neg=neg[(neg.gsw<=50)].copy()
neg['block']=(neg.x//3000).astype(int).astype(str)+'_'+(neg.y//3000).astype(int).astype(str)
print('matched 2015 negatives', len(neg), ' target', int(pc.sum()))
m=pd.concat([pos[pos.flood==1][['x','y','flood','event_year','elevation','slope','twi','dist_water_m','block']], neg[['x','y','flood','event_year','elevation','slope','twi','dist_water_m','block']]], ignore_index=True)
m.to_csv('matched_2015.csv',index=False)
print(m.groupby('flood').elevation.describe().round(2).to_string())
