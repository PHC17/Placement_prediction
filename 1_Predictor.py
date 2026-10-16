"""Live Placement Predictor page."""
import time

import plotly.graph_objects as go
import streamlit as st

from utils import theme
from utils.predictor import (BRANCHES, DEFAULTS, NUMERIC_SPEC, SAMPLES, TIERS,
                             load_artifacts, predict, validate)

st.set_page_config(page_title="Live Predictor · Placement AI", page_icon="🚀", layout="wide")
theme.sidebar()
theme.apply_theme()

META = load_artifacts()["meta"]
BASE = META["class_balance"]["Placed"] / META["n_students"]   # majority-class baseline accuracy
KEY = lambda name: f"in_{name}"          # widget key for every input


# ----------------------------------------------------------------- state helpers
def _coerce(name, value):
    """Cast to the widget's numeric type (float vs int) to avoid Streamlit type errors."""
    if name in NUMERIC_SPEC:
        step = NUMERIC_SPEC[name][4]
        return float(value) if isinstance(step, float) else int(value)
    return value


def set_form(values: dict):
    """Callback: write a full profile into widget state and clear any old result."""
    for name, v in values.items():
        st.session_state[KEY(name)] = _coerce(name, v)
    st.session_state.pop("result", None)
    st.session_state.pop("result_inp", None)


for _name, _v in DEFAULTS.items():       # initialise once per session
    st.session_state.setdefault(KEY(_name), _coerce(_name, _v))


def num(name, col, slider=False):
    """Render one numeric input from NUMERIC_SPEC into the given column."""
    label, lo, hi, _, step = NUMERIC_SPEC[name]
    lo, hi = _coerce(name, lo), _coerce(name, hi)
    with col:
        fn = st.slider if slider else st.number_input
        return fn(label, min_value=lo, max_value=hi, step=_coerce(name, step), key=KEY(name))


# ----------------------------------------------------------------- header + helpers
st.markdown('<h1 style="margin-bottom:.2rem">🚀 Live Placement Predictor</h1>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Enter a student\'s profile (information available <i>before</i> placement '
            'season) or load a sample to see the model in action.</div>', unsafe_allow_html=True)

b = st.columns([1.4, 1.4, 1.4, 1, 3])
for col, (label, vals) in zip(b[:3], SAMPLES.items()):
    col.button(label, on_click=set_form, args=(vals,), **theme.BTN, key=f"sample_{label}")
b[3].button("↺ Reset", on_click=set_form, args=(DEFAULTS,), **theme.BTN, key="reset")

# ----------------------------------------------------------------- the form
inp = {}
with st.container(border=True):
    st.markdown('<div class="form-section">🎓 Academic</div>', unsafe_allow_html=True)
    c = st.columns(5)
    inp["cgpa"] = num("cgpa", c[0])
    with c[1]:
        inp["branch"] = st.selectbox("Branch", BRANCHES, key=KEY("branch"))
    with c[2]:
        inp["college_tier"] = st.selectbox("College tier", TIERS, key=KEY("college_tier"))
    inp["attendance_percentage"] = num("attendance_percentage", c[3])
    inp["backlogs"] = num("backlogs", c[4])

with st.container(border=True):
    st.markdown('<div class="form-section">💻 Skills</div>', unsafe_allow_html=True)
    c = st.columns(4)
    for n, col in zip(["coding_skill_score", "aptitude_score", "communication_skill_score",
                       "logical_reasoning_score"], c):
        inp[n] = num(n, col, slider=True)

with st.container(border=True):
    st.markdown('<div class="form-section">🧭 Experience</div>', unsafe_allow_html=True)
    c = st.columns(3)
    for n, col in zip(["internships_count", "projects_count", "certifications_count"], c):
        inp[n] = num(n, col)
    c = st.columns(3)
    for n, col in zip(["hackathons_participated", "github_repos", "linkedin_connections"], c):
        inp[n] = num(n, col)

with st.container(border=True):
    st.markdown('<div class="form-section">🌱 Soft skills &amp; lifestyle</div>', unsafe_allow_html=True)
    c = st.columns(3)
    for n, col in zip(["mock_interview_score", "extracurricular_score", "leadership_score"], c):
        inp[n] = num(n, col, slider=True)
    c = st.columns(3)
    with c[0]:
        inp["volunteer_experience"] = st.radio("Volunteer experience", ["No", "Yes"], horizontal=True,
                                               key=KEY("volunteer_experience"))
    inp["sleep_hours"] = num("sleep_hours", c[1])
    inp["study_hours_per_day"] = num("study_hours_per_day", c[2])

go_predict = st.button("🔮 Predict Placement", type="primary", **theme.BTN)

# ----------------------------------------------------------------- predict
if go_predict:
    errors, warns = validate(inp)
    if errors:
        for e in errors:
            st.error(e, icon="🚫")
        st.session_state.pop("result", None)
    else:
        for w in warns:
            st.warning(w, icon="⚠️")
        with st.spinner("Analysing student profile…"):
            time.sleep(0.6)                       # brief pause so the loading state is visible
            try:
                st.session_state["result"] = predict(inp)
                st.session_state["result_inp"] = dict(inp)
            except Exception as exc:              # keep the UI friendly if artefacts are missing/mismatched
                st.error(f"Prediction failed: {exc}. Make sure `python train.py` has been run.", icon="🚫")

# ----------------------------------------------------------------- results
res = st.session_state.get("result")
if res:
    st.markdown('<div class="section-title">Prediction result</div>', unsafe_allow_html=True)
    if st.session_state.get("result_inp") != inp:
        st.info("Inputs have changed since this prediction — press **Predict Placement** to refresh.", icon="ℹ️")

    p = res["p_placed"]
    placed = res["label"] == "Placed"
    left, mid, right = st.columns([1.1, 1.2, 1.3])

    with left:
        st.markdown(f"""<div class="result-badge {'badge-placed' if placed else 'badge-not'}">
            <div class="small">Predicted outcome</div>
            <div class="big">{'✅ Placed' if placed else '❌ Not Placed'}</div>
            <div class="small">P(Placed) = {p*100:.1f}%</div></div>""", unsafe_allow_html=True)
        st.write("")
        st.markdown(f'<span class="risk {res["risk_css"]}">{res["risk_emoji"]} {res["risk"]}</span>',
                    unsafe_allow_html=True)
        st.write("")
        st.progress(min(max(p, 0.0), 1.0), text=f"Placement probability: {p*100:.1f}%")
        st.caption(f"Not-placed probability: {res['p_not']*100:.1f}%")

    with mid:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number={"suffix": "%", "font": {"size": 34}},
            title={"text": "Placement probability"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2563EB"},
                   "steps": [{"range": [0, META["risk_thresholds"]["low"] * 100], "color": "#FECACA"},
                             {"range": [META["risk_thresholds"]["low"] * 100,
                                        META["risk_thresholds"]["high"] * 100], "color": "#FDE68A"},
                             {"range": [META["risk_thresholds"]["high"] * 100, 100], "color": "#BBF7D0"}],
                   "threshold": {"line": {"color": "#0F172A", "width": 3}, "value": META["decision_threshold"] * 100}}))
        st.plotly_chart(theme.style_fig(gauge, 280), **theme.PLOT)
        st.caption("Bands: red = Low, amber = Moderate, green = High. Black line = decision threshold (" + f"{META['decision_threshold']*100:.1f}%).")

    with right:
        c = res["contrib"]
        top = c.reindex(c.abs().sort_values(ascending=False).head(8).index).sort_values()
        fig = go.Figure(go.Bar(x=top.values, y=[n.replace("_", " ").title() for n in top.index],
                               orientation="h",
                               marker_color=["#EF4444" if v < 0 else "#0EA5A4" for v in top.values]))
        fig.update_layout(title="Factors pushing the estimate up / down", xaxis_title="Relative influence")
        st.plotly_chart(theme.style_fig(fig, 280), **theme.PLOT)
        st.caption("Indicative influence from a linear surrogate, relative to an average student.")

    st.markdown('<div class="section-title" style="margin-top:1rem">💡 Personalised recommendations</div>',
                unsafe_allow_html=True)
    if res["weak"]:
        for _, name, advice, val, ref in res["weak"]:
            ref_txt = f"{ref:.0f}" if ref >= 10 else f"{ref:.1f}"
            val_txt = f"{val:.0f}" if val >= 10 or float(val).is_integer() else f"{val:.1f}"
            st.markdown(f'<div class="rec"><b>{name}</b> '
                        f'<span>(yours: {val_txt} · typical placed student: {ref_txt})</span><br>'
                        f'<span>{advice}</span></div>', unsafe_allow_html=True)
    else:
        st.success("No clear weak areas compared with typical placed students — keep up the good work!")
    if res["strong"]:
        st.markdown("**Strengths:** " + " ".join(f'<span class="badge">{s}</span>' for s in res["strong"]),
                    unsafe_allow_html=True)

    m = META["final_metrics"]
    st.markdown(f'<div class="note" style="margin-top:1rem">ℹ️ <b>Read with care.</b> On unseen test data this model '
                f'reaches {m["Accuracy"]*100:.1f}% accuracy (ROC-AUC {m["ROC-AUC"]:.2f}), so its probabilities are '
                f'only a mild signal — always guessing "Placed" would already score {BASE*100:.1f}%. Recommendations are analytical insights, not guaranteed outcomes.</div>',
                unsafe_allow_html=True)

theme.footer()
