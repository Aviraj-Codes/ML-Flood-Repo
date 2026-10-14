import numpy as np, pandas as pd, warnings, geopandas as gpd
warnings.filterwarnings('ignore')
from rasterio.transform import from_origin, rowcol
from scipy import ndimage
from pyproj import Transformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
rng=np.random.default_rng(11)
nrsc=gpd.read_parquet('/mnt/user-data/uploads/nrsc_flood_extent_2015.parquet').to_crs(32644)
minx,miny,maxx,maxy=nrsc.total_bounds
res=30; minx=np.floor(minx/res)*res; maxy=np.ceil(maxy/res)*res
tf30=from_origin(minx,maxy,res,res)
n1=np.load('nrsc_1.npy').astype(bool); n13=np.load('nrsc_13.npy').astype(bool)
U=np.load('flood1124.npy').astype(bool)
pre=np.load('pre_s1.npy').astype(bool)|np.load('pre_ls8.npy').astype(bool)
n13d=ndimage.binary_dilation(n13,iterations=10)
n1d=ndimage.binary_dilation(n1,iterations=10)
bg=~(n13d|n1d|U|pre)
def thin(mask,k):
    ys,xs=np.nonzero(mask)
    df=pd.DataFrame({'y':ys,'x':xs,'cell':(ys//10)*100000+(xs//10)})
    df=df.groupby('cell').sample(n=1,random_state=int(rng.integers(1e9)))
    if len(df)>k: df=df.sample(n=k,random_state=0)
    return df[['y','x']].to_numpy()
def to_xy(arr):
    return minx+(arr[:,1]+0.5)*res, maxy-(arr[:,0]+0.5)*res
pos=thin(n13,4000); neg=thin(bg,4000)
px,py=to_xy(pos); nx,ny=to_xy(neg)
pts=pd.concat([pd.DataFrame({'x':px,'y':py,'flood':1,'event_year':2015}),
               pd.DataFrame({'x':nx,'y':ny,'flood':0,'event_year':2015})],ignore_index=True)
dem=np.load('dem.npy'); slope=np.load('slope_deg.npy'); dist=np.load('dist_water.npy'); occ=np.load('occ.npy')
r,c=rowcol(from_origin(315210,1482030,30,30),pts.x.values,pts.y.values); r=np.array(r); c=np.array(c)
ok=(r>=0)&(r<dem.shape[0])&(c>=0)&(c<dem.shape[1])
pts=pts[ok].reset_index(drop=True); r=r[ok]; c=c[ok]
pts['elevation']=dem[r,c]; pts['slope']=slope[r,c]; pts['dist_water_m']=dist[r,c]; pts['gsw']=occ[r,c]
twi=np.load('twi90.npy'); r9,c9=rowcol(from_origin(315210,1482030,90,90),pts.x.values,pts.y.values)
r9=np.array(r9); c9=np.array(c9)
ok9=(r9>=0)&(r9<twi.shape[0])&(c9>=0)&(c9<twi.shape[1])
pts=pts[ok9].reset_index(drop=True); r9=r9[ok9]; c9=c9[ok9]
pts['twi']=twi[r9,c9]
wet=(pts.elevation==0)|(pts.gsw>50)|pts[['elevation','slope','twi','dist_water_m']].isna().any(axis=1)
print('wet/NaN dropped per class:', pts[wet].groupby('flood').size().to_dict())
pts=pts[~wet].reset_index(drop=True)
pts['block']=(pts.x//3000).astype(int).astype(str)+'_'+(pts.y//3000).astype(int).astype(str)
print('2015 points', pts.groupby('flood').size().to_dict())
pts.to_csv('code13_2015_points.csv',index=False)
