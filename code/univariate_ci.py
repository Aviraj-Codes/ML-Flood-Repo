import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(7)
m=pd.read_csv('cma_matched_2015.csv'); t=pd.read_csv('cma_matched_2005.csv')
m['block']=(m.x//3000).astype(int).astype(str)+'_'+(m.y//3000).astype(int).astype(str)
t['urban']=t['urban'] if 'urban' in t else t['is_urban']
B=1000; blocks=m.block.unique(); rows=[]
for c,label in [('elevation','Elevation'),('slope','Slope'),('twi','TWI'),('dist_water_m','Distance to permanent water'),('urban','Built-up cover')]:
    pt=roc_auc_score(m.flood,m[c]); bs=[]
    for _ in range(B):
        sel=rng.choice(blocks,size=len(blocks),replace=True)
        idx=np.concatenate([np.where(m.block.values==s)[0] for s in sel])
        y=m.flood.values[idx]
        if y.min()==y.max(): continue
        bs.append(roc_auc_score(y,m[c].values[idx]))
    lo,hi=np.percentile(bs,[2.5,97.5])
    pt5=roc_auc_score(t.flood,t[c]); bs5=[]
    for _ in range(B):
        idx=rng.integers(0,len(t),len(t)); y=t.flood.values[idx]
        if y.min()==y.max(): continue
        bs5.append(roc_auc_score(y,t[c].values[idx]))
    lo5,hi5=np.percentile(bs5,[2.5,97.5])
    rows.append(dict(predictor=label,auc2015=round(pt,3),ci2015_low=round(lo,3),ci2015_high=round(hi,3),auc2005=round(pt5,3),ci2005_low=round(lo5,3),ci2005_high=round(hi5,3)))
R=pd.DataFrame(rows); R.to_csv('univariate_auc_ci.csv',index=False); print(R.to_string(index=False))
