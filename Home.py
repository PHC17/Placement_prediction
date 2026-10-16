"""Home / landing page — run the app with:  streamlit run Home.py"""
import json
from pathlib import Path

import streamlit as st

from utils import theme

st.set_page_config(page_title="Student Placement Prediction System", page_icon="🎓", layout="wide")
theme.sidebar()
theme.apply_theme()

META = json.loads((Path(__file__).parent / "models" / "meta.json").read_text())
acc = META["final_metrics"]["Accuracy"] * 100

# ------------------------------------------------------------------ Hero
st.markdown("""
<div class="hero">
  <div class="pill">AI-powered · End-to-end Machine Learning project</div>
  <h1>Student Placement Prediction System</h1>
  <p>Estimate a student's placement outlook from academics, skills and career preparation —
  and get clear, actionable guidance on where to improve.</p>
</div>
""", unsafe_allow_html=True)

st.write("")
c1, c2, _ = st.columns([1.1, 1.1, 3])
with c1:
    st.page_link("pages/1_Predictor.py", label="🚀 Try Live Predictor", **theme.BTN)
with c2:
    st.page_link("pages/2_Insights.py", label="📊 View Insights", **theme.BTN)

# ------------------------------------------------------------------ Objective
st.markdown('<div class="section-title">Why this project?</div>', unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    st.markdown(theme.card("🎯", "Objective",
        "Build a classifier that predicts whether a student will be <b>Placed</b> or <b>Not Placed</b> "
        "using information available <i>before</i> the placement process."), unsafe_allow_html=True)
with right:
    st.markdown(theme.card("❓", "Problem statement",
        "Placement cells need to spot at-risk students early. This system turns profile data into a "
        "probability, a risk band and focused improvement areas — as decision support, not a verdict."),
        unsafe_allow_html=True)

# ------------------------------------------------------------------ Stats
st.markdown('<div class="section-title">At a glance</div>', unsafe_allow_html=True)
s = st.columns(4)
stats = [(f"{META['n_students']:,}", "Student records"),
         (f"{acc:.1f}%", f"Test accuracy ({META['best_model']})"),
         (str(META["n_features"]), "Input features used"),
         ("4", "Models compared")]
for col, (n, l) in zip(s, stats):
    col.markdown(theme.stat(n, l), unsafe_allow_html=True)
st.caption(f"Measured on a stratified 20% hold-out set. For context, always guessing “Placed” scores "
           f"{META['class_balance']['Placed']/META['n_students']*100:.1f}% — the data carries only a weak signal (see Insights).")

# ------------------------------------------------------------------ Feature categories
st.markdown('<div class="section-title">What the model looks at</div>', unsafe_allow_html=True)
cats = [("🎓", "Academic", "CGPA, attendance, backlogs, branch and college tier."),
        ("💻", "Technical skills", "Coding, aptitude and logical-reasoning scores; projects, GitHub repos."),
        ("🗣️", "Soft skills", "Communication, leadership, extracurricular activity and volunteering."),
        ("🧭", "Career preparation", "Internships, certifications, hackathons, mock interviews, LinkedIn network.")]
for col, (i, t, b) in zip(st.columns(4), cats):
    col.markdown(theme.card(i, t, b), unsafe_allow_html=True)

# ------------------------------------------------------------------ How it works
st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
steps = [("Enter profile", "Fill in the student's academic, skill and experience details."),
         ("Same pipeline", "Inputs go through the exact scaling and encoding used in training."),
         ("Predict", "The model returns Placed / Not Placed with a probability."),
         ("Interpret", "A risk band plus tailored recommendations highlight areas to improve.")]
for col, (i, (t, b)) in zip(st.columns(4), enumerate(steps, 1)):
    col.markdown(f'<div class="card"><div class="step"><div class="n">{i}</div>'
                 f'<div><h4>{t}</h4><p>{b}</p></div></div></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ Tech stack
st.markdown('<div class="section-title">Tech stack</div>', unsafe_allow_html=True)
st.markdown("".join(f'<span class="badge">{t}</span>' for t in
    ["Python", "Streamlit", "scikit-learn", "Pandas", "NumPy", "Plotly", "Joblib"]), unsafe_allow_html=True)

st.markdown('<div class="note" style="margin-top:1.2rem">⚠️ <b>Responsible use:</b> predictions are statistical '
            'estimates from historical data. They should support — never replace — human judgement, and must '
            'not be used to deny opportunities to any student.</div>', unsafe_allow_html=True)
theme.footer()
