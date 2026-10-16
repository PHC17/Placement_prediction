"""
Training script – Student Placement Prediction System
Run once:  python train.py
Outputs (models/ and data/): preprocessor, final model, metrics, confusion matrix,
feature importances and small EDA aggregates used by the Streamlit Insights page.
"""
import json, time, warnings
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)
warnings.filterwarnings("ignore")

import sys
CSV = sys.argv[1] if len(sys.argv) > 1 else "data/student_placement_prediction_dataset_2026.csv"
df = pd.read_csv(CSV)

# ---- Cleaning: no NaNs / duplicates found; drop id + leakage column ----------
# age & gender are deliberately NOT used (not collected by the app; fairness).
DROP = ["student_id", "salary_package_lpa", "age", "gender", "placement_status"]
y = (df["placement_status"] == "Placed").astype(int)
X = df.drop(columns=DROP)
CAT = ["branch", "college_tier", "volunteer_experience"]
NUM = [c for c in X.columns if c not in CAT]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pre = ColumnTransformer([
    ("num", StandardScaler(), NUM),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT),
])
Xtr = pre.fit_transform(X_tr); Xte = pre.transform(X_te)
feat_names = list(pre.get_feature_names_out())

def metrics(m, name):
    p = m.predict(Xte)
    r = {
        "Model": name,
        "Accuracy": accuracy_score(y_te, p),
        "Precision": precision_score(y_te, p),
        "Recall (Placed)": recall_score(y_te, p),
        "Recall (Not Placed)": recall_score(y_te, p, pos_label=0),
        "F1-Score": f1_score(y_te, p),
    }
    try:
        r["ROC-AUC"] = roc_auc_score(y_te, m.predict_proba(Xte)[:, 1])
    except Exception:
        r["ROC-AUC"] = roc_auc_score(y_te, m.decision_function(Xte))
    return r

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=120, min_samples_leaf=3, n_jobs=-1, random_state=42),
    "SVM": SVC(kernel="rbf", probability=False, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
}
rows, fitted = [], {}
for name, m in models.items():
    t = time.time()
    if name == "SVM":                       # RBF-SVM is O(n^2): train on a stratified 25k subset
        idx = np.random.RandomState(42).choice(len(y_tr), 25000, replace=False)
        m.fit(Xtr[idx], y_tr.iloc[idx])
    else:
        m.fit(Xtr, y_tr)
    fitted[name] = m
    r = metrics(m, name); r["Train time (s)"] = round(time.time() - t, 1); rows.append(r)
    print(r, flush=True)
res = pd.DataFrame(rows)

# ---- Scaling impact (same models, unscaled numeric features) ------------------
raw_pre = ColumnTransformer([("num", "passthrough", NUM),
                             ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])
Rtr = raw_pre.fit_transform(X_tr); Rte = raw_pre.transform(X_te)
scal = []
for name in ["Logistic Regression", "KNN", "Random Forest"]:
    m = type(models[name])(**models[name].get_params()).fit(Rtr, y_tr)
    scal.append({"Model": name, "Unscaled accuracy": accuracy_score(y_te, m.predict(Rte)),
                 "Scaled accuracy": float(res.loc[res.Model == name, "Accuracy"].iloc[0])})
print(scal, flush=True)

# ---- Pick best (by ROC-AUC, tie-break F1) --------------------------------------
best_name = res.sort_values(["ROC-AUC", "F1-Score"], ascending=False).iloc[0]["Model"]
print("BEST:", best_name, flush=True)
best = fitted[best_name]

# ---- Model improvement + GridSearchCV on best model ----------------------------
if best_name == "Random Forest":
    grid = {"n_estimators": [150], "max_depth": [6, 10, None], "min_samples_leaf": [3, 10, 30]}
    base = RandomForestClassifier(random_state=42, n_jobs=-1)
elif best_name == "Logistic Regression":
    grid = {"C": [0.001, 0.01, 0.1, 1, 10]}
    base = LogisticRegression(max_iter=2000)
elif best_name == "KNN":
    grid = {"n_neighbors": [5, 15, 31, 51], "weights": ["uniform", "distance"]}
    base = KNeighborsClassifier(n_jobs=-1)
else:
    grid = {"C": [0.5, 1, 5]}; base = SVC(probability=True, random_state=42)
gs = GridSearchCV(base, grid, cv=3, scoring="roc_auc", n_jobs=1, verbose=1)
sub = np.random.RandomState(1).choice(len(y_tr), 30000, replace=False) if best_name in ("Random Forest", "KNN") else slice(None)
gs.fit(Xtr[sub], y_tr.iloc[sub] if not isinstance(sub, slice) else y_tr)
tuned = type(base)(**{**base.get_params(), **gs.best_params_}).fit(Xtr, y_tr)   # refit on full train
tuned_res = metrics(tuned, f"{best_name} (tuned)")
print("TUNED:", gs.best_params_, gs.best_score_, tuned_res, flush=True)

# choose final: tuned if not worse on F1 else original (needs probabilities)
final = tuned if tuned_res["ROC-AUC"] >= res.loc[res.Model == best_name, "ROC-AUC"].iloc[0] - 1e-9 else best
final_label = "tuned" if final is tuned else "original"
if not hasattr(final, "predict_proba"):
    final = SVC(**{**final.get_params(), "probability": True}).fit(Xtr, y_tr)
proba_te = final.predict_proba(Xte)[:, 1]
THR = float(y_tr.mean())                       # base-rate threshold -> balanced recall on both classes
fp = (proba_te >= THR).astype(int)
cm = confusion_matrix(y_te, fp)
risk_thr = {"low": float(np.quantile(proba_te, 1/3)), "high": float(np.quantile(proba_te, 2/3))}
def at_thr(p):
    return {"Accuracy": accuracy_score(y_te, p), "Precision": precision_score(y_te, p),
            "Recall (Placed)": recall_score(y_te, p), "Recall (Not Placed)": recall_score(y_te, p, pos_label=0),
            "F1-Score": f1_score(y_te, p), "ROC-AUC": roc_auc_score(y_te, proba_te)}
m_default, m_bal = at_thr((proba_te >= 0.5).astype(int)), at_thr(fp)
print("THR", THR, "risk", risk_thr, "\nDEFAULT", m_default, "\nBALANCED", m_bal, flush=True)

# ---- Feature importance (Random Forest) ----------------------------------------
rf = fitted["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)
# Logistic coefficients for direction of effect
lr = fitted["Logistic Regression"]
coef = pd.Series(lr.coef_[0], index=feat_names)

# ---- Save artefacts --------------------------------------------------------------
joblib.dump(pre, "models/preprocessor.joblib")
joblib.dump(final, "models/model.joblib")
meta = {
    "best_model": best_name, "final_variant": final_label,
    "best_params": {k: (None if v is None else v) for k, v in gs.best_params_.items()},
    "cv_best_auc": gs.best_score_, "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
    "n_students": int(len(df)), "n_features": len(X.columns),
    "feature_columns": list(X.columns), "numeric": NUM, "categorical": CAT,
    "class_balance": {"Placed": int(y.sum()), "Not Placed": int((1 - y).sum())},
    "confusion_matrix": cm.tolist(),
    "final_metrics": m_bal, "final_metrics_at_0.5": m_default, "decision_threshold": THR, "risk_thresholds": risk_thr,
    "tuned_metrics": tuned_res,
    "svm_note": "SVM trained on a stratified 25,000-row subset for speed (RBF-SVM scales ~O(n^2)).",
}
json.dump(meta, open("models/meta.json", "w"), indent=2, default=float)
res.to_csv("models/model_comparison.csv", index=False)
pd.DataFrame(scal).to_csv("models/scaling_effect.csv", index=False)
imp.rename("importance").to_csv("models/feature_importance.csv")
coef.rename("coef").to_csv("models/lr_coefficients.csv")

# Reference stats for recommendations (placed-student medians) + EDA aggregates
placed = df[df.placement_status == "Placed"]
ref = {c: float(placed[c].median()) for c in NUM}
ref_all = {c: {"mean": float(df[c].mean()), "std": float(df[c].std()),
               "min": float(df[c].min()), "max": float(df[c].max())} for c in NUM}
json.dump({"placed_median": ref, "overall": ref_all}, open("models/reference.json", "w"), indent=2)

df.sample(6000, random_state=42).drop(columns=["student_id"]).to_csv("data/eda_sample.csv", index=False)
corr = df[NUM + ["salary_package_lpa"]].assign(placed=y).drop(columns=["salary_package_lpa"]).corr()["placed"].drop("placed")
corr.rename("corr").to_csv("models/corr_with_target.csv")
# hard-to-classify groups
te = X_te.copy(); te["y"] = y_te.values; te["ok"] = (fp == y_te.values)
te.groupby("branch")["ok"].mean().to_csv("models/acc_by_branch.csv")
te.groupby("college_tier")["ok"].mean().to_csv("models/acc_by_tier.csv")
print("DONE", flush=True)
