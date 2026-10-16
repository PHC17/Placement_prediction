"""Shared look & feel: CSS, light/dark toggle, Plotly styling, small HTML helpers."""
import inspect

import streamlit as st


def _stretch(fn) -> dict:
    """Full-width kwarg that works on old (use_container_width) and new (width='stretch') Streamlit."""
    try:
        if "width" in inspect.signature(fn).parameters:
            return {"width": "stretch"}
    except (TypeError, ValueError):
        pass
    return {"use_container_width": True}


BTN, PLOT, DF = _stretch(st.button), _stretch(st.plotly_chart), _stretch(st.dataframe)

PALETTES = {
    "light": dict(bg="#F5F8FC", surface="#FFFFFF", text="#0F172A", muted="#64748B",
                  border="#E2E8F0", primary="#2563EB", accent="#0EA5A4",
                  shadow="0 4px 18px rgba(15,23,42,.07)", soft="#EFF6FF"),
    "dark":  dict(bg="#0B1220", surface="#111A2E", text="#E6EDF7", muted="#94A3B8",
                  border="#22304A", primary="#60A5FA", accent="#2DD4BF",
                  shadow="0 4px 22px rgba(0,0,0,.45)", soft="#13213B"),
}


def is_dark() -> bool:
    return bool(st.session_state.get("dark_mode", False))


def palette() -> dict:
    return PALETTES["dark" if is_dark() else "light"]


def style_fig(fig, height=360):
    """Apply consistent transparent styling to any Plotly figure."""
    p = palette()
    fig.update_layout(template="plotly_dark" if is_dark() else "plotly_white", height=height,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=p["text"], family="Inter, sans-serif"),
                      margin=dict(l=10, r=10, t=50, b=10),
                      title=dict(font=dict(size=16)))
    return fig


_BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root { --bg:%(bg)s; --surface:%(surface)s; --text:%(text)s; --muted:%(muted)s;
  --border:%(border)s; --primary:%(primary)s; --accent:%(accent)s; --shadow:%(shadow)s; --soft:%(soft)s; }
html, body, [class*="css"], .stApp { font-family:'Inter',sans-serif; }
.stApp { background:var(--bg); color:var(--text); }
header[data-testid="stHeader"] { background:transparent; }
.block-container { padding-top:2rem; padding-bottom:3rem; max-width:1150px; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp label { color:var(--text); }
[data-testid="stSidebar"] { background:var(--surface); border-right:1px solid var(--border); }

/* Hero */
.hero { background:linear-gradient(135deg,#1D4ED8 0%%,#2563EB 45%%,#0EA5A4 100%%);
  border-radius:24px; padding:3.2rem 2.4rem; box-shadow:var(--shadow); }
.hero h1 { color:#fff !important; font-size:2.7rem; font-weight:800; margin:0 0 .6rem 0; line-height:1.15; }
.hero p  { color:rgba(255,255,255,.92) !important; font-size:1.12rem; max-width:720px; margin:0; }
.hero .pill { display:inline-block; background:rgba(255,255,255,.18); color:#fff; padding:.3rem .8rem;
  border-radius:999px; font-size:.8rem; font-weight:600; margin-bottom:1rem; }

/* Cards */
.card { background:var(--surface); border:1px solid var(--border); border-radius:18px;
  padding:1.3rem 1.4rem; box-shadow:var(--shadow); height:100%%; }
.card h4 { margin:.2rem 0 .4rem 0; font-size:1.05rem; font-weight:700; color:var(--text); }
.card p, .card li { color:var(--muted); font-size:.93rem; margin:0; }
.card .icon { font-size:1.7rem; }
.stat .num { font-size:2rem; font-weight:800; color:var(--primary); line-height:1.1; }
.stat .lbl { color:var(--muted); font-size:.85rem; font-weight:500; margin-top:.25rem; }
.section-title { font-size:1.5rem; font-weight:800; margin:2.2rem 0 .3rem 0; color:var(--text); }
.section-sub { color:var(--muted); margin-bottom:1.1rem; }
.badge { display:inline-block; background:var(--soft); color:var(--primary); border:1px solid var(--border);
  padding:.3rem .75rem; border-radius:999px; font-size:.82rem; font-weight:600; margin:.2rem .25rem .2rem 0; }
.step { display:flex; gap:.9rem; align-items:flex-start; }
.step .n { min-width:2.1rem; height:2.1rem; border-radius:50%%; background:var(--primary); color:#fff;
  display:flex; align-items:center; justify-content:center; font-weight:700; }

/* Predictor */
.form-section { font-weight:700; font-size:1.05rem; margin:.2rem 0 .6rem 0; color:var(--primary); }
.result-badge { border-radius:20px; padding:1.6rem; text-align:center; box-shadow:var(--shadow); }
.result-badge .big { font-size:2.2rem; font-weight:800; color:#fff; }
.result-badge .small { color:rgba(255,255,255,.9); font-size:.95rem; }
.badge-placed { background:linear-gradient(135deg,#059669,#0EA5A4); }
.badge-not    { background:linear-gradient(135deg,#DC2626,#F97316); }
.risk { display:inline-block; padding:.45rem 1rem; border-radius:999px; font-weight:700; font-size:.95rem; }
.risk-high { background:#DCFCE7; color:#166534; }
.risk-mod  { background:#FEF3C7; color:#92400E; }
.risk-low  { background:#FEE2E2; color:#991B1B; }
.rec { border-left:4px solid var(--accent); background:var(--soft); border-radius:10px;
  padding:.8rem 1rem; margin-bottom:.6rem; }
.rec b { color:var(--text); }
.rec span { color:var(--muted); font-size:.92rem; }
.note { background:var(--soft); border:1px dashed var(--border); border-radius:12px;
  padding:.8rem 1rem; color:var(--muted); font-size:.88rem; }

/* Widgets */
.stButton > button { border-radius:12px; font-weight:600; padding:.55rem 1.2rem;
  border:1px solid var(--border); transition:all .15s ease; }
.stButton > button:hover { transform:translateY(-1px); border-color:var(--accent); }
.stButton > button[kind="primary"] { background:linear-gradient(135deg,#2563EB,#0EA5A4); color:#fff; border:none; }
.stButton > button[kind="primary"]:hover { color:#fff; box-shadow:0 6px 18px rgba(37,99,235,.35); }
[data-testid="stPageLink-NavLink"] { background:var(--surface); border:1px solid var(--border); border-radius:12px; }
[data-testid="stExpander"] { background:var(--surface); border:1px solid var(--border); border-radius:14px; }
.footer { text-align:center; color:var(--muted); font-size:.85rem; margin-top:3rem;
  padding-top:1.2rem; border-top:1px solid var(--border); }
@media (max-width: 640px) { .hero { padding:2rem 1.3rem; } .hero h1 { font-size:2rem; } }
"""

_DARK_CSS = """
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="base-input"], .stNumberInput input {
  background-color:#0B1220 !important; color:%(text)s !important; border-color:%(border)s !important; }
[data-baseweb="popover"] *, [role="listbox"] { background:%(surface)s !important; color:%(text)s !important; }
.stButton > button { background:%(surface)s; color:%(text)s; }
.stTabs [data-baseweb="tab"] { color:%(muted)s; }
"""


def apply_theme():
    """Inject CSS for the current mode. Call once at the top of every page."""
    p = palette()
    css = _BASE_CSS % p + (_DARK_CSS % p if is_dark() else "")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def sidebar():
    """Sidebar branding + dark-mode toggle (state persists across pages)."""
    with st.sidebar:
        st.markdown("### 🎓 PlacementAI")
        st.caption("Student Placement Prediction System")
        st.toggle("🌙 Dark mode", key="dark_mode")


def card(icon: str, title: str, body: str) -> str:
    return f'<div class="card"><div class="icon">{icon}</div><h4>{title}</h4><p>{body}</p></div>'


def stat(num: str, label: str) -> str:
    return f'<div class="card stat"><div class="num">{num}</div><div class="lbl">{label}</div></div>'


def footer():
    st.markdown('<div class="footer">Student Placement Prediction System · Built with Streamlit, '
                'scikit-learn &amp; Plotly · Educational decision-support only.</div>',
                unsafe_allow_html=True)
