import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
exec(open('elev_reversal_check.py').read().split('for width in')[0])
res=[]
for seed in range(50):
    rng=np.random.default_rng(seed)
    m=match(pool,1.0)
    raw=roc_auc_score(m.flood,m.elevation)   # single-variable AUC, no model
    oof=np.zeros(len(m)); fold_auc=[]; slopes=[]
    for a,b in GroupKFold(5).split(m,m.flood,groups=m.block):
        mm=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
        mm.fit(m.iloc[a][['elevation']],m.iloc[a].flood)
        p=mm.predict_proba(m.iloc[b][['elevation']])[:,1]; oof[b]=p
        fold_auc.append(roc_auc_score(m.iloc[b].flood,p)); slopes.append(mm[-1].coef_[0][0])
    res.append((raw,roc_auc_score(m.flood,oof),np.mean(fold_auc),np.mean(np.sign(slopes)>0)))
R=np.array(res)
print('raw elevation AUC (no model): mean %.3f range %.3f-%.3f' % (R[:,0].mean(),R[:,0].min(),R[:,0].max()))
print('pooled OOF AUC (as reported): mean %.3f range %.3f-%.3f' % (R[:,1].mean(),R[:,1].min(),R[:,1].max()))
print('mean within-fold AUC: mean %.3f' % R[:,2].mean())
print('share of folds with positive elevation coefficient: %.2f' % R[:,3].mean())
