import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from rasterio.transform import from_origin, rowcol
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from pyproj import Transformer
tr=pd.read_csv('code13_2015_points.csv')
d05=pd.read_csv('final_predictors.csv')
te=d05[d05.event_year==2005].copy()
te=te.rename(columns={'slope':'slope'})
hot=pd.read_csv('gcc_hotspots_2015.csv')
t=Transformer.from_crs(4326,32644,always_xy=True)
hx,hy=t.transform(hot.longitude.astype(float).values, hot.latitude.astype(float).values)
dem=np.load('dem.npy'); slope=np.load('slope_deg.npy'); dist=np.load('dist_water.npy'); twi=np.load('twi90.npy')
r,c=rowcol(from_origin(315210,1482030,30,30),hx,hy); r=np.array(r); c=np.array(c)
r9,c9=rowcol(from_origin(315210,1482030,90,90),hx,hy); r9=np.array(r9); c9=np.array(c9)
ok=(r>=0)&(r<dem.shape[0])&(c>=0)&(c<dem.shape[1])&(r9>=0)&(r9<twi.shape[0])&(c9>=0)&(c9<twi.shape[1])
hot_f=pd.DataFrame({'elevation':dem[r,c],'slope':slope[r,c],'dist_water_m':dist[r,c],'twi':twi[r9,c9]})[ok].dropna()
hot_f['flood']=1
rng=np.random.default_rng(5)
rb=rng.integers(0,dem.shape[0],20000); cb=rng.integers(0,dem.shape[1],20000)
bgf=pd.DataFrame({'elevation':dem[rb,cb],'slope':slope[rb,cb],'dist_water_m':dist[rb,cb],'twi':twi[np.minimum(rb//3,twi.shape[0]-1),np.minimum(cb//3,twi.shape[1]-1)]}).dropna()
bgf=bgf[bgf.elevation!=0].head(3000)
bgf['flood']=0
COLS_ALL=['elevation','slope','twi','dist_water_m']
def model(name):
    if name=='logreg': return make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
    if name=='rf': return RandomForestClassifier(n_estimators=300,min_samples_leaf=5,class_weight='balanced',n_jobs=-1,random_state=0)
def fit(name,X,y):
    m=CalibratedClassifierCV(model(name),method='isotonic',cv=3); m.fit(X,y); return m
print('== spatial CV within 2015 (code13 labels)')
for cols,label in [(COLS_ALL,'all four'),(['elevation'],'elevation only')]:
    for name in ['logreg','rf']:
        oof=np.zeros(len(tr))
        for a,b in GroupKFold(5).split(tr,tr.flood,groups=tr.block):
            m=fit(name,tr.iloc[a][cols],tr.iloc[a].flood); oof[b]=m.predict_proba(tr.iloc[b][cols])[:,1]
        print(f'  {label:15s} {name:6s} AUC {roc_auc_score(tr.flood,oof):.3f}  Brier {brier_score_loss(tr.flood,oof):.3f}')
print('== transfer: train 2015 code13, test IRS 2005 (same 2005 negatives/positives as before; urban flag dropped)')
for cols,label in [(COLS_ALL,'all four'),(['elevation'],'elevation only')]:
    for name in ['logreg','rf']:
        m=fit(name,tr[cols],tr.flood); p=m.predict_proba(te[cols])[:,1]
        print(f'  {label:15s} {name:6s} AUC {roc_auc_score(te.flood,p):.3f}  Brier {brier_score_loss(te.flood,p):.3f}')
print('== independent check: GCC hotspots (positives) vs random background (negatives), model trained on 2015 code13')
hb=pd.concat([hot_f,bgf],ignore_index=True)
for cols,label in [(COLS_ALL,'all four'),(['elevation'],'elevation only')]:
    for name in ['logreg','rf']:
        m=fit(name,tr[cols],tr.flood); p=m.predict_proba(hb[cols])[:,1]
        print(f'  {label:15s} {name:6s} hotspot-vs-background AUC {roc_auc_score(hb.flood,p):.3f}  (n hot {len(hot_f)})')
print('2015 code13 positives median elevation', round(tr[tr.flood==1].elevation.median(),2), ' negatives', round(tr[tr.flood==0].elevation.median(),2))
