import numpy as np, pandas as pd, rasterio, geopandas as gpd, warnings
warnings.filterwarnings('ignore')
from rasterio.transform import rowcol
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(12)
cma=gpd.read_file('CMA.geojson').to_crs(32644).geometry.union_all()
with rasterio.open('/mnt/user-data/uploads/chennai_worldcover_v200_32644.tif') as s:
    lc=s.read(1); tfw=s.transform
def add_lc(df):
    r,c=rowcol(tfw,df.x.values,df.y.values); r=np.array(r); c=np.array(c)
    ok=(r>=0)&(r<lc.shape[0])&(c>=0)&(c<lc.shape[1])
    v=np.full(len(df),-1); v[ok]=lc[r[ok],c[ok]]
    df=df.copy(); df['urban']=(v==50).astype(int); return df
def in_cma(df):
    g=gpd.GeoSeries(gpd.points_from_xy(df.x,df.y),crs=32644)
    return g.within(cma).values
# ---- 2015: primary code-13 design
p15=pd.read_csv('code13_2015_points.csv')
p15=add_lc(p15); p15=p15[in_cma(p15)].reset_index(drop=True)
print('2015 inside CMA', p15.groupby('flood').size().to_dict())
bins=np.arange(0,31,1)
p15['bin']=np.digitize(p15.elevation.clip(0,29.999),bins)-1
pc=p15[p15.flood==1].groupby('bin').size(); nc=p15[p15.flood==0].groupby('bin').size()
keep=pd.concat([pc,nc],axis=1,keys=['p','n']).fillna(0); keep['k']=keep[['p','n']].min(axis=1).astype(int)
parts=[]
for b,row in keep.iterrows():
    k=int(row.k)
    if k==0: continue
    for f in [0,1]:
        idx=p15.index[(p15.flood==f)&(p15.bin==b)]
        parts.append(p15.loc[rng.choice(idx,size=k,replace=False)])
m15=pd.concat(parts,ignore_index=True)
print('2015 matched in CMA', m15.groupby('flood').size().to_dict(), ' elev mean', m15.groupby('flood').elevation.mean().round(2).to_dict())
# ---- 2005: IRS design, inside CMA, matched 0-30 m
d05=pd.read_csv('final_predictors.csv'); t=d05[d05.event_year==2005].copy()
t=t[in_cma(t)].reset_index(drop=True); t['urban']=t['is_urban']
t['bin']=np.digitize(t.elevation.clip(0,29.999),bins)-1
t=t[(t.elevation>=0)&(t.elevation<30)].reset_index(drop=True)
pc=t[t.flood==1].groupby('bin').size(); nc=t[t.flood==0].groupby('bin').size()
keep=pd.concat([pc,nc],axis=1,keys=['p','n']).fillna(0); keep['k']=keep[['p','n']].min(axis=1).astype(int)
parts=[]
for b,row in keep.iterrows():
    k=int(row.k)
    if k==0: continue
    for f in [0,1]:
        idx=t.index[(t.flood==f)&(t.bin==b)]
        parts.append(t.loc[rng.choice(idx,size=k,replace=False)])
m05=pd.concat(parts,ignore_index=True)
print('2005 inside CMA, 0-30 m, matched', m05.groupby('flood').size().to_dict(), ' elev mean', m05.groupby('flood').elevation.mean().round(2).to_dict())
m15.to_csv('cma_matched_2015.csv',index=False); m05.to_csv('cma_matched_2005.csv',index=False)
def mdl(n):
    return make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced')) if n=='logreg' else RandomForestClassifier(n_estimators=300,min_samples_leaf=5,class_weight='balanced',n_jobs=-1,random_state=0)
sets=[(['elevation'],'elevation only'),(['slope','twi','dist_water_m'],'terrain, no elevation'),(['urban'],'built-up only'),(['slope','twi','dist_water_m','urban'],'terrain + built-up'),(['elevation','slope','twi','dist_water_m','urban'],'all five')]
print('== matched 2015 spatial CV, inside CMA')
for cols,label in sets:
    out=[]
    for n in ['logreg','rf']:
        oof=np.zeros(len(m15))
        for a,b in GroupKFold(5).split(m15,m15.flood,groups=m15.block):
            mm=mdl(n); mm.fit(m15.iloc[a][cols],m15.iloc[a].flood); oof[b]=mm.predict_proba(m15.iloc[b][cols])[:,1]
        out.append(roc_auc_score(m15.flood,oof))
    print(f'  {label:24s} logreg {out[0]:.3f}  rf {out[1]:.3f}')
print('== matched transfer: train matched 2015 (CMA), test matched 2005 (CMA)')
for cols,label in sets:
    out=[]
    for n in ['logreg','rf']:
        mm=mdl(n); mm.fit(m15[cols],m15.flood); out.append(roc_auc_score(m05.flood,mm.predict_proba(m05[cols])[:,1]))
    print(f'  {label:24s} logreg {out[0]:.3f}  rf {out[1]:.3f}')
