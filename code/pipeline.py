import sys
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss, confusion_matrix
from xgboost import XGBClassifier
import shap

PREDICTORS = ["elevation", "slope", "twi", "is_urban", "dist_water_m"]
TARGET = "flood"
EVENT = "event_year"
BLOCK = "block_id"
COORDS = ["x", "y"]

def load(path):
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    missing = [c for c in PREDICTORS + [TARGET, EVENT] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = df.dropna(subset=PREDICTORS + [TARGET]).reset_index(drop=True)
    return df

def make_block_ids(df, size_m):
    bx = (df["x"] // size_m).astype(int)
    by = (df["y"] // size_m).astype(int)
    df[BLOCK] = bx.astype(str) + "_" + by.astype(str)
    return df

def models():
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=0),
        "xgb": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=0),
    }

def metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

def spatial_cv(train, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for name in models().keys():
        oof = np.zeros(len(train))
        for tr, va in gkf.split(train, train[TARGET], groups=train[BLOCK]):
            m = models()[name]
            m_cal = CalibratedClassifierCV(m, method="isotonic", cv=3)
            m_cal.fit(train.iloc[tr][PREDICTORS], train.iloc[tr][TARGET])
            oof[va] = m_cal.predict_proba(train.iloc[va][PREDICTORS])[:, 1]
        r = metrics(train[TARGET].values, oof)
        r["model"] = name
        rows.append(r)
    return pd.DataFrame(rows)

def transfer(train, test):
    rows = []
    fitted = {}
    for name in models().keys():
        m_cal = CalibratedClassifierCV(models()[name], method="isotonic", cv=3)
        m_cal.fit(train[PREDICTORS], train[TARGET])
        p = m_cal.predict_proba(test[PREDICTORS])[:, 1]
        r = metrics(test[TARGET].values, p)
        r["model"] = name
        rows.append(r)
        fitted[name] = m_cal
    return pd.DataFrame(rows), fitted

def shap_report(train, test, out_prefix):
    x = models()["xgb"]
    x.fit(train[PREDICTORS], train[TARGET])
    explainer = shap.TreeExplainer(x)
    sv = explainer.shap_values(test[PREDICTORS])
    imp = pd.Series(np.abs(sv).mean(axis=0), index=PREDICTORS).sort_values(ascending=False)
    imp.to_csv(f"{out_prefix}_shap_importance.csv", header=["mean_abs_shap"])
    corr = train[PREDICTORS].corr()
    corr.to_csv(f"{out_prefix}_predictor_corr.csv")
    return imp

def main(data_path, block_m, out_prefix):
    df = load(data_path)
    df = make_block_ids(df, block_m)
    train = df[df[EVENT] == 2015].reset_index(drop=True)
    test = df[df[EVENT] == 2005].reset_index(drop=True)
    if len(train) == 0 or len(test) == 0:
        raise ValueError("need rows for event_year 2015 and 2005")
    cv = spatial_cv(train)
    cv.to_csv(f"{out_prefix}_spatial_cv_2015.csv", index=False)
    tr, _ = transfer(train, test)
    tr.to_csv(f"{out_prefix}_transfer_2015_to_2005.csv", index=False)
    imp = shap_report(train, test, out_prefix)
    print("spatial CV (2015):")
    print(cv.round(3).to_string(index=False))
    print("transfer 2015 -> 2005:")
    print(tr.round(3).to_string(index=False))
    print("SHAP importance:")
    print(imp.round(4).to_string())

if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), sys.argv[3])
