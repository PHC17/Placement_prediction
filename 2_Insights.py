"""Insights & Model Performance page — all numbers are read from artefacts produced by train.py."""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import theme

st.set_page_config(page_title="Insights · Placement AI", page_icon="📊", layout="wide")
theme.sidebar()
theme.apply_theme()

ROOT = Path(__file__).resolve().parent.parent
M, D = ROOT / "models", ROOT / "data"
BLUE, TEAL, RED = "#2563EB", "#0EA5A4", "#EF4444"


@st.cache_data
def load():
    meta = json.loads((M / "meta.json").read_text())
    return dict(
        meta=meta,
        comp=pd.read_csv(M / "model_comparison.csv"),
        scal=pd.read_csv(M / "scaling_effect.csv"),
        imp=pd.read_csv(M / "feature_importance.csv", index_col=0)["importance"],
        corr=pd.read_csv(M / "corr_with_target.csv", index_col=0)["corr"],
        by_branch=pd.read_csv(M / "acc_by_branch.csv", index_col=0).iloc[:, 0],
        by_tier=pd.read_csv(M / "acc_by_tier.csv", index_col=0).iloc[:, 0],
        eda=pd.read_csv(D / "eda_sample.csv"),
    )


d = load()
meta, comp, eda = d["meta"], d["comp"], d["eda"]
eda["placed"] = (eda["placement_status"] == "Placed").astype(int)
BASE = meta["class_balance"]["Placed"] / meta["n_students"]
fm = meta["final_metrics"]
pretty = lambda s: s.replace("num__", "").replace("cat__", "").replace("_", " ").title()


def rate_bar(col, title, order=None, bins=None, labels=None):
    """Placement-rate bar chart for a categorical column or a binned numeric column."""
    x = eda[col] if bins is None else pd.cut(eda[col], bins=bins, labels=labels)
    g = eda.groupby(x, observed=True)["placed"].agg(["mean", "size"]).reset_index()
    g.columns = [col, "rate", "n"]
    g["rate"] *= 100
    if order:
        g[col] = pd.Categorical(g[col], order, ordered=True)
        g = g.sort_values(col)
    fig = px.bar(g, x=col, y="rate", text=g["rate"].round(1).astype(str) + "%", hover_data={"n": True},
                 color_discrete_sequence=[BLUE], title=title)
    fig.add_hline(y=BASE * 100, line_dash="dash", line_color=TEAL, annotation_text="overall rate")
    fig.update_layout(yaxis_title="Placement rate (%)", xaxis_title=None, yaxis_range=[0, 100])
    return theme.style_fig(fig, 340)


# ------------------------------------------------------------------ header
st.markdown('<h1 style="margin-bottom:.2rem">📊 Insights &amp; Model Performance</h1>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">What the data says, how the models compare, and how far the '
            'results can be trusted.</div>', unsafe_allow_html=True)

k = st.columns(4)
for col, (n, l) in zip(k, [(f"{meta['n_students']:,}", "Students analysed"),
                           (f"{fm['Accuracy']*100:.1f}%", "Final test accuracy"),
                           (f"{BASE*100:.1f}%", "Baseline: always predict “Placed”"),
                           (f"{fm['ROC-AUC']:.3f}", "ROC-AUC (0.5 = random)")]):
    col.markdown(theme.stat(n, l), unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3, tab4 = st.tabs(["🔎 Data insights", "🤖 Model comparison", "🧩 Errors & drivers", "📘 Method & limits"])

# ================================================================== TAB 1 – EDA
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        cb = meta["class_balance"]
        fig = px.pie(names=list(cb.keys()), values=list(cb.values()), hole=.6, title="Placement outcome balance",
                     color=list(cb.keys()), color_discrete_map={"Placed": TEAL, "Not Placed": RED})
        st.plotly_chart(theme.style_fig(fig, 340), **theme.PLOT)
    with c2:
        c = d["corr"].sort_values()
        fig = go.Figure(go.Bar(x=c.values, y=[pretty(i) for i in c.index], orientation="h",
                               marker_color=[RED if v < 0 else BLUE for v in c.values]))
        fig.update_layout(title="Correlation of each feature with placement", xaxis_title="Pearson r")
        st.plotly_chart(theme.style_fig(fig, 340), **theme.PLOT)

    top = d["corr"].abs().sort_values(ascending=False)
    st.markdown(f'<div class="note">📌 <b>Takeaway:</b> the classes are only mildly imbalanced '
                f'({BASE*100:.1f}% placed). The strongest linear relationships with placement are '
                f'<b>{pretty(top.index[0])}</b> (r = {d["corr"][top.index[0]]:+.3f}), '
                f'<b>{pretty(top.index[1])}</b> ({d["corr"][top.index[1]]:+.3f}) and '
                f'<b>{pretty(top.index[2])}</b> ({d["corr"][top.index[2]]:+.3f}). '
                f'Correlation is association, not causation.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Placement rate by group</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    a.plotly_chart(rate_bar("cgpa", "By CGPA band", bins=[0, 6, 7, 8, 9, 10.01],
                            labels=["<6", "6–7", "7–8", "8–9", "9–10"]), **theme.PLOT)
    b.plotly_chart(rate_bar("internships_count", "By number of internships"), **theme.PLOT)
    a, b = st.columns(2)
    a.plotly_chart(rate_bar("college_tier", "By college tier", order=["Tier 1", "Tier 2", "Tier 3"]),
                   **theme.PLOT)
    b.plotly_chart(rate_bar("branch", "By branch"), **theme.PLOT)
    a, b = st.columns(2)
    a.plotly_chart(rate_bar("backlogs", "By number of backlogs"), **theme.PLOT)
    b.plotly_chart(rate_bar("volunteer_experience", "By volunteer experience"), **theme.PLOT)

    st.markdown('<div class="section-title">Explore any feature</div>', unsafe_allow_html=True)
    num_cols = [c for c in meta["numeric"]]
    feat = st.selectbox("Feature", num_cols, index=num_cols.index("coding_skill_score"), format_func=pretty)
    fig = px.histogram(eda, x=feat, color="placement_status", barmode="overlay", opacity=.65, nbins=40,
                       histnorm="probability density", color_discrete_map={"Placed": TEAL, "Not Placed": RED},
                       title=f"{pretty(feat)} — Placed vs Not Placed")
    st.plotly_chart(theme.style_fig(fig, 360), **theme.PLOT)
    st.caption("Charts use a random 6,000-student sample of the 100,000 records for speed.")

# ================================================================== TAB 2 – MODELS
with tab2:
    st.markdown('<div class="section-title" style="margin-top:.5rem">Model comparison (20% hold-out, default 0.5 threshold)</div>',
                unsafe_allow_html=True)
    show = comp[["Model", "Accuracy", "Precision", "Recall (Placed)", "Recall (Not Placed)", "F1-Score", "ROC-AUC"]]
    st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns[1:]})
                 .highlight_max(subset=show.columns[1:], color="#BBF7D0" if not theme.is_dark() else "#14532D"),
                 **theme.DF, hide_index=True)
    long = show.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall (Placed)", "F1-Score"])
    fig = px.bar(long, x="Model", y="value", color="variable", barmode="group", title="Metric comparison",
                 color_discrete_sequence=[BLUE, TEAL, "#7C3AED", "#F59E0B"])
    fig.add_hline(y=BASE, line_dash="dash", annotation_text="majority-class accuracy")
    fig.update_layout(yaxis_title=None, yaxis_range=[0, 1], legend_title=None)
    st.plotly_chart(theme.style_fig(fig, 380), **theme.PLOT)

    best = meta["best_model"]
    st.markdown(f'<div class="note">🏆 <b>Selected model: {best}.</b> The four models are close together, so the '
                f'choice used <b>ROC-AUC</b> (threshold-independent probability quality) — important because the app '
                f'shows probabilities and risk bands. Accuracy and F1 differences between the top models are within '
                f'sampling noise. {meta["svm_note"]}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Why accuracy alone is not enough</div>', unsafe_allow_html=True)
    st.markdown(f"At the default 0.5 cut-off every model finds most *placed* students but misses most *not-placed* "
                f"ones (Not-Placed recall ≈ {comp['Recall (Not Placed)'].min()*100:.0f}–"
                f"{comp['Recall (Not Placed)'].max()*100:.0f}%). Since a placement cell mainly wants to catch "
                f"at-risk students, the app uses a **balanced decision threshold of "
                f"{meta['decision_threshold']*100:.1f}%** (the training placement rate).")
    t1, t2 = st.columns(2)
    d0, d1 = meta["final_metrics_at_0.5"], meta["final_metrics"]
    tab = pd.DataFrame({"Threshold 0.50": d0, f"Balanced {meta['decision_threshold']:.3f}": d1}).T
    t1.dataframe(tab.style.format("{:.3f}"), **theme.DF)
    t2.markdown(f'<div class="note">Moving the threshold trades some Placed-recall for much better Not-Placed '
                f'recall: <b>{d0["Recall (Not Placed)"]*100:.0f}% → {d1["Recall (Not Placed)"]*100:.0f}%</b>.</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-title">Does feature scaling matter?</div>', unsafe_allow_html=True)
    sc = d["scal"].melt(id_vars="Model", var_name="Setting", value_name="Accuracy")
    fig = px.bar(sc, x="Model", y="Accuracy", color="Setting", barmode="group",
                 color_discrete_sequence=["#94A3B8", BLUE], title="Test accuracy: unscaled vs scaled inputs")
    fig.update_layout(yaxis_range=[.45, .62])
    st.plotly_chart(theme.style_fig(fig, 320), **theme.PLOT)
    st.caption("Distance- and gradient-based models (KNN, Logistic Regression, SVM) are scale-sensitive; "
               "tree ensembles such as Random Forest are not.")

    st.markdown('<div class="section-title">Hyperparameter tuning</div>', unsafe_allow_html=True)
    orig = comp.loc[comp.Model == best].iloc[0]
    tm = meta["tuned_metrics"]
    cmp_t = pd.DataFrame({"Original": orig[["Accuracy", "Precision", "Recall (Placed)", "Recall (Not Placed)", "F1-Score", "ROC-AUC"]].astype(float),
                          "Tuned": pd.Series({k: tm[k] for k in ["Accuracy", "Precision", "Recall (Placed)", "Recall (Not Placed)", "F1-Score", "ROC-AUC"]})}).T
    x1, x2 = st.columns([1.4, 1])
    x1.dataframe(cmp_t.style.format("{:.4f}"), **theme.DF)
    x2.markdown(f'<div class="card"><h4>GridSearchCV (3-fold)</h4><p>Best params: <code>{meta["best_params"]}</code>'
                f'<br>Best CV ROC-AUC: <b>{meta["cv_best_auc"]:.4f}</b><br>Final variant used: <b>{meta["final_variant"]}</b></p></div>',
                unsafe_allow_html=True)
    st.caption("Tuning changes results only marginally — consistent with a model limited by the information in the "
               "features, not by its settings. More complex models do not guarantee better real-world performance.")

# ================================================================== TAB 3 – CONFUSION + IMPORTANCE
with tab3:
    c1, c2 = st.columns(2)
    cm = meta["confusion_matrix"]
    with c1:
        fig = px.imshow(cm, text_auto=True, x=["Pred: Not Placed", "Pred: Placed"], y=["Actual: Not Placed", "Actual: Placed"],
                        color_continuous_scale=["#E0F2FE", BLUE], title=f"Confusion matrix — {best} (balanced threshold)")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(theme.style_fig(fig, 380), **theme.PLOT)
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        st.markdown(f"- ✅ Placed correctly identified: **{tp:,}**  \n- ✅ Not-placed correctly identified: **{tn:,}**  \n"
                    f"- ⚠️ **False positives** (predicted Placed, actually Not Placed): **{fp:,}**  \n"
                    f"- ⚠️ **False negatives** (predicted Not Placed, actually Placed): **{fn:,}**")
        st.caption("For a placement cell, false positives are usually costlier: an at-risk student is told they are "
                   "fine and may miss support. False negatives cost extra support effort.")
    with c2:
        imp = d["imp"].head(12).sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=[pretty(i) for i in imp.index], orientation="h", marker_color=TEAL))
        fig.update_layout(title="Random Forest feature importance (top 12)", xaxis_title="Importance")
        st.plotly_chart(theme.style_fig(fig, 380), **theme.PLOT)
        spread = d["imp"].max() / max(d["imp"].min(), 1e-9)
        st.caption(f"Importances are fairly flat (top ÷ bottom ≈ {spread:.1f}×) — no single feature dominates, "
                   "matching the weak correlations in the data insights tab.")

    st.markdown('<div class="section-title">Which students are hardest to classify?</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    for col, ser, ttl in [(a, d["by_branch"], "Test accuracy by branch"), (b, d["by_tier"], "Test accuracy by college tier")]:
        fig = px.bar(x=ser.index, y=ser.values * 100, title=ttl, color_discrete_sequence=[BLUE],
                     text=[f"{v*100:.1f}%" for v in ser.values])
        fig.update_layout(yaxis_title="Accuracy (%)", xaxis_title=None, yaxis_range=[40, 70])
        col.plotly_chart(theme.style_fig(fig, 300), **theme.PLOT)

# ================================================================== TAB 4 – METHOD & LIMITS
with tab4:
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"""<div class="card"><h4>🛠️ Methodology</h4><ul>
        <li><b>Data:</b> {meta['n_students']:,} students, no missing values or duplicates found.</li>
        <li><b>Removed:</b> <code>student_id</code> (identifier), <code>salary_package_lpa</code> (post-placement → data leakage), plus <code>age</code> and <code>gender</code> (not collected by the app; fairness).</li>
        <li><b>Features:</b> {meta['n_features']} inputs — scaled numerics (StandardScaler) and one-hot categoricals.</li>
        <li><b>Split:</b> stratified 80/20, <code>random_state=42</code> ({meta['n_train']:,} train / {meta['n_test']:,} test).</li>
        <li><b>Models:</b> Logistic Regression, Random Forest, SVM, KNN → best by ROC-AUC → GridSearchCV tuning.</li>
        <li><b>Risk bands:</b> Low / Moderate / High = tertiles of predicted probability on the test set.</li></ul></div>""",
                    unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="card"><h4>⚠️ Limitations &amp; ethical notes</h4><ul>
        <li><b>Weak signal:</b> accuracy of {fm['Accuracy']*100:.1f}% vs a {BASE*100:.1f}% baseline — predictions are only mildly better than guessing; treat them as one input among many.</li>
        <li><b>Historical data:</b> past patterns may not hold for future hiring markets, companies or cohorts.</li>
        <li><b>Missing drivers:</b> real placement depends on openings, company criteria and luck — not captured here.</li>
        <li><b>Correlation ≠ causation:</b> improving a feature is not guaranteed to change an outcome.</li>
        <li><b>Fairness:</b> branch and college tier are included; monitor for group disparities and never use the score to deny opportunities.</li>
        <li><b>Privacy:</b> student data is sensitive — handle it with consent and minimal retention.</li></ul></div>""",
                    unsafe_allow_html=True)

theme.footer()
