"""
Prediction engine: loads the saved preprocessor + model, validates input,
predicts, classifies risk and builds explanations / recommendations.
Kept free of Streamlit UI code so it can be unit-tested on its own.
"""
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

BRANCHES = ["CSE", "IT", "ECE", "EEE", "Mechanical", "Civil"]
TIERS = ["Tier 1", "Tier 2", "Tier 3"]

# name -> (label, min, max, default, step)  — ranges follow the training data
NUMERIC_SPEC = {
    "cgpa": ("CGPA (out of 10)", 0.0, 10.0, 7.5, 0.01),
    "attendance_percentage": ("Attendance (%)", 0.0, 100.0, 85.0, 0.5),
    "backlogs": ("Active backlogs", 0, 10, 0, 1),
    "coding_skill_score": ("Coding skill score", 0.0, 100.0, 70.0, 1.0),
    "aptitude_score": ("Aptitude score", 0.0, 100.0, 65.0, 1.0),
    "communication_skill_score": ("Communication skill score", 0.0, 100.0, 68.0, 1.0),
    "logical_reasoning_score": ("Logical reasoning score", 0.0, 100.0, 66.0, 1.0),
    "internships_count": ("Internships", 0, 10, 1, 1),
    "projects_count": ("Projects", 0, 20, 3, 1),
    "certifications_count": ("Certifications", 0, 15, 2, 1),
    "hackathons_participated": ("Hackathons participated", 0, 15, 1, 1),
    "github_repos": ("GitHub repositories", 0, 50, 4, 1),
    "linkedin_connections": ("LinkedIn connections", 0, 5000, 500, 10),
    "mock_interview_score": ("Mock interview score", 0.0, 100.0, 70.0, 1.0),
    "extracurricular_score": ("Extracurricular score", 0.0, 100.0, 60.0, 1.0),
    "leadership_score": ("Leadership score", 0.0, 100.0, 55.0, 1.0),
    "sleep_hours": ("Sleep (hours / day)", 0.0, 14.0, 7.0, 0.1),
    "study_hours_per_day": ("Study (hours / day)", 0.0, 16.0, 3.5, 0.1),
}
DEFAULTS = {k: v[3] for k, v in NUMERIC_SPEC.items()}
DEFAULTS.update(branch="CSE", college_tier="Tier 2", volunteer_experience="No")

# Three demo profiles for the "Load sample student" buttons
SAMPLES = {
    "🌟 Strong candidate": dict(
        cgpa=8.9, branch="CSE", college_tier="Tier 1", attendance_percentage=94, backlogs=0,
        coding_skill_score=92, aptitude_score=84, communication_skill_score=82,
        logical_reasoning_score=85, internships_count=3, projects_count=6,
        certifications_count=4, hackathons_participated=3, github_repos=12,
        linkedin_connections=850, mock_interview_score=88, extracurricular_score=75,
        leadership_score=72, volunteer_experience="Yes", sleep_hours=7.5, study_hours_per_day=5.0),
    "⚖️ Average student": dict(DEFAULTS),
    "⚠️ Needs support": dict(
        cgpa=6.0, branch="Mechanical", college_tier="Tier 3", attendance_percentage=68, backlogs=3,
        coding_skill_score=42, aptitude_score=45, communication_skill_score=50,
        logical_reasoning_score=48, internships_count=0, projects_count=1,
        certifications_count=0, hackathons_participated=0, github_repos=1,
        linkedin_connections=120, mock_interview_score=48, extracurricular_score=40,
        leadership_score=35, volunteer_experience="No", sleep_hours=5.5, study_hours_per_day=1.5),
}

# Recommendation text per feature: (friendly name, higher_is_better, advice)
ADVICE = {
    "cgpa": ("CGPA", True, "Aim to lift your CGPA through consistent coursework and revision; many recruiters shortlist on a CGPA cut-off."),
    "backlogs": ("Backlogs", False, "Clear pending backlogs as a priority — they are a common eligibility filter for campus drives."),
    "attendance_percentage": ("Attendance", True, "Improve attendance; it often reflects discipline and is checked for eligibility."),
    "coding_skill_score": ("Coding skills", True, "Practise data structures & algorithms regularly (e.g. 1–2 problems a day) and build a small portfolio."),
    "aptitude_score": ("Aptitude", True, "Take timed quantitative and verbal aptitude mock tests weekly to build speed and accuracy."),
    "communication_skill_score": ("Communication", True, "Practise structured speaking — group discussions, presentations and writing clear emails."),
    "logical_reasoning_score": ("Logical reasoning", True, "Work on puzzles and reasoning sets; focus on patterns, series and data interpretation."),
    "internships_count": ("Internships", True, "Look for internships, even short or remote ones — practical exposure strengthens a profile."),
    "projects_count": ("Projects", True, "Build 2–3 end-to-end projects and document them on GitHub with clear READMEs."),
    "certifications_count": ("Certifications", True, "Add one or two relevant, skill-focused certifications aligned with your target roles."),
    "hackathons_participated": ("Hackathons", True, "Join a hackathon to demonstrate teamwork and problem-solving under pressure."),
    "github_repos": ("GitHub activity", True, "Keep your GitHub active with well-described repositories."),
    "linkedin_connections": ("Professional network", True, "Grow your LinkedIn network with alumni and recruiters, and keep your profile complete."),
    "mock_interview_score": ("Interview preparation", True, "Schedule regular mock interviews and review feedback on technical and HR rounds."),
    "extracurricular_score": ("Extracurriculars", True, "Take part in clubs or events — they show initiative beyond academics."),
    "leadership_score": ("Leadership", True, "Take a lead role in a club, project or event team."),
    "study_hours_per_day": ("Study routine", True, "Build a steady daily study routine and protect it with a timetable."),
    "sleep_hours": ("Sleep", True, "Keep sleep in a healthy 7–8 hour range; it supports focus and learning."),
}


# --------------------------------------------------------------------- loading
@lru_cache(maxsize=1)
def load_artifacts() -> dict:
    """Load model, preprocessor and reference files once per process."""
    return {
        "model": joblib.load(MODELS / "model.joblib"),
        "pre": joblib.load(MODELS / "preprocessor.joblib"),
        "meta": json.loads((MODELS / "meta.json").read_text()),
        "ref": json.loads((MODELS / "reference.json").read_text()),
        "coef": pd.read_csv(MODELS / "lr_coefficients.csv", index_col=0)["coef"],
    }


# ------------------------------------------------------------------ validation
def validate(inp: dict):
    """Return (errors, warnings). Errors block prediction; warnings do not."""
    errors, warns = [], []
    for k, (label, lo, hi, _, _) in NUMERIC_SPEC.items():
        v = inp.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            errors.append(f"**{label}** is required.")
        elif not (lo <= v <= hi):
            errors.append(f"**{label}** must be between {lo} and {hi}.")
    if inp.get("branch") not in BRANCHES:
        errors.append("Please choose a valid **branch**.")
    if inp.get("college_tier") not in TIERS:
        errors.append("Please choose a valid **college tier**.")
    if not errors:
        if inp["sleep_hours"] + inp["study_hours_per_day"] > 24:
            errors.append("Sleep + study hours cannot exceed **24 hours** per day.")
        if inp["sleep_hours"] < 4:
            warns.append("Sleep below 4 hours/day is unusually low — please double-check.")
        if inp["study_hours_per_day"] > 12:
            warns.append("Study above 12 hours/day is unusually high — please double-check.")
        if inp["cgpa"] < 4:
            warns.append("CGPA below 4.0 is outside the typical range of the training data.")
    return errors, warns


# ------------------------------------------------------------------- prediction
def to_frame(inp: dict) -> pd.DataFrame:
    """Build a 1-row frame with the exact training column order."""
    cols = load_artifacts()["meta"]["feature_columns"]
    return pd.DataFrame([{c: inp[c] for c in cols}])


def risk_category(p_placed: float, thresholds: dict):
    if p_placed >= thresholds["high"]:
        return "High Placement Probability", "risk-high", "🟢"
    if p_placed >= thresholds["low"]:
        return "Moderate Placement Probability", "risk-mod", "🟡"
    return "Low Placement Probability", "risk-low", "🔴"


def contributions(X_t, feat_names, coef) -> pd.Series:
    """Signed per-feature contribution from a linear surrogate (coef × scaled value).
    One-hot columns are summed back to their parent feature."""
    vec = np.asarray(X_t.todense() if hasattr(X_t, "todense") else X_t).ravel()
    s = pd.Series(vec * coef.reindex(feat_names).values, index=feat_names)
    grouped = {}
    for name, val in s.items():
        base = name.split("__", 1)[1]
        for cat in ("branch", "college_tier", "volunteer_experience"):
            if base.startswith(cat):
                base = cat
        grouped[base] = grouped.get(base, 0.0) + val
    return pd.Series(grouped).sort_values()


def recommendations(inp: dict, contrib: pd.Series, max_items=5):
    """Rank weak areas by (gap to placed-student median, in std units) × model weight."""
    art = load_artifacts()
    ref, ov, coef = art["ref"]["placed_median"], art["ref"]["overall"], art["coef"]
    weights = coef.abs()
    weak, strong = [], []
    for k, (name, higher_better, advice) in ADVICE.items():
        sd = ov[k]["std"] or 1.0
        gap = (ref[k] - inp[k]) / sd if higher_better else (inp[k] - ref[k]) / sd
        if k == "backlogs":
            gap = float(inp[k] > 0) * (1 + inp[k] / 3)
        w = float(weights.get(f"num__{k}", weights.mean()))
        if gap > 0.25:
            weak.append((gap * (w + 0.02), name, advice, inp[k], ref[k]))
        elif gap < -0.5:
            strong.append(name)
    weak.sort(reverse=True)
    return weak[:max_items], strong[:5]


def predict(inp: dict) -> dict:
    art = load_artifacts()
    X = to_frame(inp)
    Xt = art["pre"].transform(X)
    proba = float(art["model"].predict_proba(Xt)[0, 1])
    thr = art["meta"]["decision_threshold"]          # balanced (base-rate) threshold, see Insights page
    label = "Placed" if proba >= thr else "Not Placed"
    cat, css, emoji = risk_category(proba, art["meta"]["risk_thresholds"])
    contrib = contributions(Xt, list(art["pre"].get_feature_names_out()), art["coef"])
    weak, strong = recommendations(inp, contrib)
    return dict(label=label, p_placed=proba, p_not=1 - proba, risk=cat, risk_css=css,
                risk_emoji=emoji, contrib=contrib, weak=weak, strong=strong)
