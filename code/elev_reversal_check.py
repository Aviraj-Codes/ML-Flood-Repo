import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import geopandas as gpd
from shapely.geometry import Point
cma=gpd.read_file('CMA.geojson').to_crs(32644).geometry.union_all()
pool=pd.read_csv('code13_2015_points.csv')
g=gpd.GeoSeries(gpd.points_from_xy(pool.x,pool.y),crs=32644)
pool=pool[g.within(cma).values].reset_index(drop=True)
pool['block']=(pool.x//3000).astype(int).astype(str)+'_'+(pool.y//3000).astype(int).astype(str)
rng=np.random.default_rng(0)
def match(df,width,seeds=50):
    lo,hi=0,30
    bins=np.arange(lo,hi+width,width)
    df=df[(df.elevation>=lo)&(df.elevation<hi)].copy()
    df['bin']=np.floor((df.elevation-lo)/width).astype(int)
    pc=df[df.flood==1].groupby('bin').size(); nc=df[df.flood==0].groupby('bin').size()
    k=pd.concat([pc,nc],axis=1,keys=['p','n']).fillna(0)
    k=k.min(axis=1,numeric_only=False) if False else np.minimum(k.p,k.n).astype(int)
    out=[]
    for b,kk in k.items():
        if kk==0: continue
        for f in [0,1]:
            idx=df.index[(df.flood==f)&(df.bin==b)]
            out.append(df.loc[rng.choice(idx,size=kk,replace=False)])
    return pd.concat(out,ignore_index=True)
for width in [1.0,0.5,0.25]:
    aucs=[];within=[];n=[]
    for seed in range(50):
        rng=np.random.default_rng(seed)
        m=match(pool,width)
        n.append(int((m.flood==1).sum()))
        # elevation-only logistic, 5-fold grouped CV
        oof=np.zeros(len(m))
        for a,b in GroupKFold(5).split(m,m.flood,groups=m.block):
            mm=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
            mm.fit(m.iloc[a][['elevation']],m.iloc[a].flood); oof[b]=mm.predict_proba(m.iloc[b][['elevation']])[:,1]
        aucs.append(roc_auc_score(m.flood,oof))
        # within-bin elevation difference (flood minus non-flood), averaged over bins
        bins=np.floor(m.elevation/width).astype(int)
        d=[]
        for bb in np.unique(bins):
            s=m[bins==bb]
            if (s.flood==1).any() and (s.flood==0).any():
                d.append(s[s.flood==1].elevation.mean()-s[s.flood==0].elevation.mean())
        within.append(np.mean(d))
    print(f'bin {width} m: n per class {int(np.mean(n))}, elevation-only AUC mean {np.mean(aucs):.3f} (range {np.min(aucs):.3f}-{np.max(aucs):.3f}), mean within-bin elevation diff (flood - non-flood) {np.mean(within):+.3f} m')
