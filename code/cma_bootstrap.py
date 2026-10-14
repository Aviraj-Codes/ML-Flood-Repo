import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
m=pd.read_csv('cma_matched_2015.csv'); t=pd.read_csv('cma_matched_2005.csv')
m['block']=(m.x//3000).astype(int).astype(str)+'_'+(m.y//3000).astype(int).astype(str)
def mdl(n):
    return make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced')) if n=='logreg' else RandomForestClassifier(n_estimators=300,min_samples_leaf=5,class_weight='balanced',n_jobs=-1,random_state=0)
sets=[(['elevation'],'elevation only'),(['slope','twi','dist_water_m'],'terrain, no elevation'),(['urban'],'built-up only'),(['slope','twi','dist_water_m','urban'],'terrain + built-up')]
rng=np.random.default_rng(99)
B=1000
blocks=m.block.unique()
res=[]
for cols,label in sets:
    for n in ['logreg','rf']:
        oof=np.zeros(len(m))
        for a,b in GroupKFold(5).split(m,m.flood,groups=m.block):
            mm=mdl(n); mm.fit(m.iloc[a][cols],m.iloc[a].flood); oof[b]=mm.predict_proba(m.iloc[b][cols])[:,1]
        point=roc_auc_score(m.flood,oof)
        boots=[]
        for _ in range(B):
            sel=rng.choice(blocks,size=len(blocks),replace=True)
            idx=np.concatenate([np.where(m.block.values==s)[0] for s in sel])
            yb=m.flood.values[idx]
            if yb.min()==yb.max(): continue
            boots.append(roc_auc_score(yb,oof[idx]))
        lo,hi=np.percentile(boots,[2.5,97.5])
        res.append(dict(test='within-2015 spatial CV',predictors=label,model=n,auc=round(point,3),ci_low=round(lo,3),ci_high=round(hi,3)))
        mm=mdl(n); mm.fit(m[cols],m.flood); p=mm.predict_proba(t[cols])[:,1]
        point=roc_auc_score(t.flood,p)
        boots=[]
        for _ in range(B):
            idx=rng.integers(0,len(t),len(t))
            yb=t.flood.values[idx]
            if yb.min()==yb.max(): continue
            boots.append(roc_auc_score(yb,p[idx]))
        lo,hi=np.percentile(boots,[2.5,97.5])
        res.append(dict(test='transfer to 2005',predictors=label,model=n,auc=round(point,3),ci_low=round(lo,3),ci_high=round(hi,3)))
R=pd.DataFrame(res)
R.to_csv('cma_bootstrap_ci.csv',index=False)
print(R.to_string(index=False))
