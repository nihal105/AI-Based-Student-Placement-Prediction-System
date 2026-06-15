import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Based Student Placement Prediction",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

* { font-family: 'Inter', sans-serif; }

html, body, .main {
    background-color: #07090F !important;
    color: #E2E8F0;
}

.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* ── HERO BANNER ── */
.hero-banner {
    position: relative;
    background: linear-gradient(135deg, #0D0F1E 0%, #111827 60%, #0D0F1E 100%);
    border: 1px solid #1E2A3D;
    border-radius: 20px;
    padding: 40px 44px 36px 44px;
    margin-bottom: 2rem;
    overflow: hidden;
}
/* animated gradient orbs */
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -80px; right: -40px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(124,58,237,0.14) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(129,140,248,0.3);
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #818CF8;
    margin-bottom: 16px;
}
.hero-dot {
    width: 7px; height: 7px;
    background: #6EE7B7;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 8px #6EE7B7;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.7); }
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: -0.04em;
    line-height: 1.1;
    margin: 0 0 10px 0;
}
.hero-accent {
    background: linear-gradient(90deg, #818CF8, #A78BFA, #7C3AED);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1rem;
    color: #475569;
    font-weight: 400;
    margin: 0 0 24px 0;
    line-height: 1.6;
}
.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 4px;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #0F1523;
    border: 1px solid #1E2A3D;
    border-radius: 8px;
    padding: 6px 13px;
    font-size: 0.78rem;
    color: #64748B;
    font-weight: 500;
}
.hero-badge span { font-size: 0.85rem; }

/* Score card */
.score-ring-wrap {
    background: #0F1523;
    border: 1px solid #1E2A3D;
    border-radius: 18px;
    padding: 22px 16px;
    text-align: center;
}
.score-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #475569;
    font-weight: 600;
    margin-bottom: 4px;
}
.score-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #F1F5F9;
}
.score-unit { font-size: 1rem; color: #475569; }

/* Sidebar */
.css-1d391kg, [data-testid="stSidebar"] {
    background-color: #0B0F1A !important;
    border-right: 1px solid #1E2A3D !important;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    color: #818CF8;
    font-family: 'Space Grotesk', sans-serif;
}

/* Predict button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    color: #F8FAFC;
    border-radius: 12px;
    height: 3.4em;
    font-size: 1rem;
    font-weight: 600;
    border: none;
    letter-spacing: 0.02em;
    transition: all 0.25s ease;
    box-shadow: 0 4px 20px rgba(79,70,229,0.35);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(124,58,237,0.45);
}

/* Result boxes */
.result-placed {
    background: linear-gradient(135deg, #022C22 0%, #064E3B 100%);
    border: 1px solid #059669;
    padding: 28px 20px;
    border-radius: 16px;
    text-align: center;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #6EE7B7;
    box-shadow: 0 0 40px rgba(5,150,105,0.15);
}
.result-notplaced {
    background: linear-gradient(135deg, #2D0A0A 0%, #7F1D1D 100%);
    border: 1px solid #DC2626;
    padding: 28px 20px;
    border-radius: 16px;
    text-align: center;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #FCA5A5;
    box-shadow: 0 0 40px rgba(220,38,38,0.15);
}

/* Rec cards */
.rec-critical {
    background: #2D0A0A;
    border-left: 4px solid #EF4444;
    padding: 14px 18px;
    border-radius: 0 10px 10px 0;
    margin: 8px 0;
    color: #FCA5A5;
    font-size: 0.92rem;
}
.rec-warning {
    background: #1C1500;
    border-left: 4px solid #F59E0B;
    padding: 14px 18px;
    border-radius: 0 10px 10px 0;
    margin: 8px 0;
    color: #FDE68A;
    font-size: 0.92rem;
}
.rec-tip {
    background: #001A15;
    border-left: 4px solid #10B981;
    padding: 14px 18px;
    border-radius: 0 10px 10px 0;
    margin: 8px 0;
    color: #6EE7B7;
    font-size: 0.92rem;
}
.rec-title {
    font-weight: 700;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
    opacity: 0.7;
}

/* Section headers */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1.8rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1E2A3D;
}

/* Profile score bar */
.profile-bar-wrap { margin-bottom: 14px; }
.profile-bar-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: #94A3B8;
    margin-bottom: 5px;
    font-weight: 500;
}
.profile-bar-bg {
    background: #1E2A3D;
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
}
.profile-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.6s ease;
}

/* Divider */
hr { border-color: #1E2A3D !important; margin: 1.5rem 0; }

/* Stmetric */
[data-testid="metric-container"] {
    background: #0F1523;
    border: 1px solid #1E2A3D;
    border-radius: 14px;
    padding: 14px 16px;
}
[data-testid="stMetricValue"] {
    color: #818CF8 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.4rem !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #334155;
    font-size: 0.82rem;
    margin-top: 2rem;
    letter-spacing: 0.04em;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("student_placement_pipeline.pkl")

try:
    pipeline = load_model()
    model_loaded = True
except:
    model_loaded = False
# ---------------------------------------------------
# HELPER: Compute a weighted placement readiness score (0-100)
# ---------------------------------------------------
def compute_readiness_score(study_hours, attendance, sleep_hours,
                             internet_usage, assignments_completed,
                             previous_score, exam_score):
    weights = {
        "exam_score":             0.28,
        "previous_score":         0.20,
        "attendance":             0.18,
        "study_hours":            0.14,
        "assignments_completed":  0.10,
        "sleep_hours":            0.06,
        "internet_usage":         0.04,
    }
    normalized = {
        "exam_score":             exam_score / 100,
        "previous_score":         previous_score / 100,
        "attendance":             attendance / 100,
        "study_hours":            min(study_hours / 10, 1.0),
        "assignments_completed":  assignments_completed / 100,
        "sleep_hours":            min(sleep_hours / 8, 1.0),
        "internet_usage":         max(0, 1 - (internet_usage / 12)),
    }
    score = sum(weights[k] * normalized[k] for k in weights)
    return round(score * 100, 1)


# ---------------------------------------------------
# HELPER: Smart AI recommendations
# ---------------------------------------------------
def get_smart_recommendations(study_hours, attendance, sleep_hours,
                               internet_usage, assignments_completed,
                               previous_score, exam_score, probability):

    recs = []  # Each rec: (level, title, message, impact)

    # --- Exam Score ---
    if exam_score < 50:
        recs.append(("critical", "Low Exam Score",
            f"Your exam score ({exam_score:.0f}) is critically low. "
            "Prioritise past paper practice, timed mock tests, and topic-by-topic revision. "
            "Improving this has the highest impact on placement odds.",
            "High Impact"))
    elif exam_score < 65:
        recs.append(("warning", "Exam Score Needs Work",
            f"An exam score of {exam_score:.0f} is below most placement cut-offs (65+). "
            "Focus on weak subjects and increase practice test frequency.",
            "High Impact"))
    elif exam_score < 80:
        recs.append(("tip", "Good Exam Score — Push Further",
            f"Your exam score ({exam_score:.0f}) is solid. "
            "Targeting 80+ will significantly improve shortlisting chances at competitive companies.",
            "Medium Impact"))

    # --- Attendance ---
    if attendance < 60:
        recs.append(("critical", "Critical Attendance",
            f"Attendance at {attendance:.0f}% is severely low. "
            "Many companies require ≥ 75% attendance as a baseline eligibility criterion. "
            "Irregular attendance also correlates strongly with lower exam performance.",
            "High Impact"))
    elif attendance < 75:
        recs.append(("warning", "Attendance Below Threshold",
            f"Attendance at {attendance:.0f}% falls below the typical 75% eligibility bar. "
            "Aim to bring this above 75% to stay eligible for campus placements.",
            "High Impact"))

    # --- Study Hours ---
    if study_hours < 2:
        recs.append(("critical", "Very Low Study Hours",
            f"Only {study_hours:.1f} hrs/day of study is insufficient for placement preparation. "
            "Students who land placements typically study 5–8 hrs/day. "
            "Build a structured daily study plan immediately.",
            "High Impact"))
    elif study_hours < 4:
        recs.append(("warning", "Study Hours Below Average",
            f"At {study_hours:.1f} hrs/day you're below the recommended 5+ hrs. "
            "Even one extra focused hour per day adds up to 30+ hours monthly.",
            "Medium Impact"))
    elif study_hours > 11:
        recs.append(("tip", "Study Hours: Check Quality vs. Quantity",
            f"You're studying {study_hours:.1f} hrs/day. Sustained high-volume study can lead to burnout. "
            "Ensure you're using active recall and spaced repetition, not passive reading.",
            "Low Impact"))

    # --- Assignments ---
    if assignments_completed < 30:
        recs.append(("critical", "Very Low Assignment Completion",
            f"Only {assignments_completed}% of assignments completed. "
            "Assignments reinforce learning and build skills directly relevant to placement tests. "
            "Prioritise clearing the backlog this week.",
            "Medium Impact"))
    elif assignments_completed < 60:
        recs.append(("warning", "Incomplete Assignment Record",
            f"At {assignments_completed}% completion, you're missing practice opportunities. "
            "Target 80%+ to ensure consistent skill reinforcement.",
            "Medium Impact"))

    # --- Sleep ---
    if sleep_hours < 5:
        recs.append(("critical", "Dangerously Low Sleep",
            f"Only {sleep_hours:.1f} hrs of sleep severely impairs memory consolidation and cognitive function. "
            "This directly reduces your ability to retain study material. Aim for 7–8 hrs.",
            "Medium Impact"))
    elif sleep_hours < 6.5:
        recs.append(("warning", "Insufficient Sleep",
            f"At {sleep_hours:.1f} hrs, you're below the recommended 7–8 hrs. "
            "Poor sleep reduces information retention and exam performance. "
            "Try shifting bedtime earlier by 30 min each week.",
            "Medium Impact"))
    elif sleep_hours > 10:
        recs.append(("warning", "Oversleeping May Be a Concern",
            f"Sleeping {sleep_hours:.1f} hrs could indicate low motivation or poor time use. "
            "Maintain a consistent schedule with 7–8 hrs for peak mental performance.",
            "Low Impact"))

    # --- Internet Usage ---
    if internet_usage > 8:
        recs.append(("critical", "Excessive Internet Usage",
            f"At {internet_usage:.1f} hrs/day, internet use is likely displacing study time. "
            "Use app timers to cap non-study screen time to under 3 hrs/day.",
            "Medium Impact"))
    elif internet_usage > 5:
        recs.append(("warning", "High Internet Usage",
            f"{internet_usage:.1f} hrs/day of internet use can fragment focus. "
            "Redirect 1–2 hrs toward coding platforms (LeetCode, HackerRank) for skill-building.",
            "Low Impact"))

    # --- Previous Score ---
    score_gap = exam_score - previous_score
    if previous_score < 55 and exam_score < 60:
        recs.append(("critical", "Consistently Low Academic Performance",
            "Both previous and current exam scores are below 60. "
            "Identify foundational gaps through a diagnostic test and address core concepts first.",
            "High Impact"))
    elif score_gap < -10:
        recs.append(("warning", "Performance Declining",
            f"Your exam score dropped {abs(score_gap):.0f} pts from your previous score. "
            "Analyse which topics you scored lower in and schedule targeted revision sessions.",
            "Medium Impact"))
    elif score_gap > 10:
        recs.append(("tip", "Strong Improvement Trend",
            f"You improved {score_gap:.0f} pts over your previous score — excellent trajectory. "
            "Maintain this momentum with consistent revision and mock tests.",
            "Positive Signal"))

    # --- Holistic / Probability-based ---
    if probability < 40 and len(recs) < 3:
        recs.append(("critical", "Overall Profile Needs Significant Work",
            "Your current profile gives a low probability of placement. "
            "Focus on exam score and attendance first — these have the highest weightage. "
            "Consider reaching out to professors for extra guidance.",
            "High Impact"))
    elif probability >= 85:
        recs.append(("tip", "Strong Placement Profile",
            "Your metrics are in excellent shape. Now focus on interview preparation: "
            "DSA practice, mock GDs, HR questions, and company-specific preparation.",
            "Next Step"))
    elif probability >= 65:
        recs.append(("tip", "Good Profile — Fine-Tune Your Weak Areas",
            "You're in a good position. Address the 1–2 weaker areas above and focus on "
            "aptitude tests and communication skills for the final push.",
            "Next Step"))

    return recs


# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown("""
<div class="hero-banner">
  <div class="hero-eyebrow">
    <span class="hero-dot"></span>
    AI-Powered &nbsp;·&nbsp; Campus Placement Intelligence
  </div>
  <h1 class="hero-title">
    Student&nbsp;<span class="hero-accent">Placement</span><br>Predictor
  </h1>
  <p class="hero-sub">
    Enter your academic profile in the left panel and get an instant AI-powered
    placement prediction with personalised improvement recommendations.
  </p>
  <div class="hero-badges">
    <div class="hero-badge"><span>🧠</span> ML Prediction Model</div>
    <div class="hero-badge"><span>📊</span> Weighted Readiness Score</div>
    <div class="hero-badge"><span>💡</span> Smart Recommendations</div>
    <div class="hero-badge"><span>⚡</span> Real-Time Analysis</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.markdown("## 📌 Student Profile")
st.sidebar.markdown("---")

study_hours         = st.sidebar.slider("📚 Study Hours / Day", 0.0, 15.0, 5.0, step=0.5)
attendance          = st.sidebar.slider("🏫 Attendance (%)", 0.0, 100.0, 75.0, step=1.0)
sleep_hours         = st.sidebar.slider("😴 Sleep Hours / Day", 0.0, 12.0, 7.0, step=0.5)
internet_usage      = st.sidebar.slider("🌐 Internet Usage (hrs/day)", 0.0, 15.0, 4.0, step=0.5)
assignments_completed = st.sidebar.slider("📝 Assignments Completed (%)", 0, 100, 50)
previous_score      = st.sidebar.slider("📊 Previous Score (0–100)", 0.0, 100.0, 60.0, step=0.5)
exam_score          = st.sidebar.slider("🎯 Exam Score (0–100)", 0.0, 100.0, 70.0, step=0.5)

# ---------------------------------------------------
# LIVE READINESS SCORE
# ---------------------------------------------------
readiness = compute_readiness_score(
    study_hours, attendance, sleep_hours,
    internet_usage, assignments_completed,
    previous_score, exam_score
)

grade = (
    ("S", "#6EE7B7", "#059669") if readiness >= 85 else
    ("A", "#93C5FD", "#3B82F6") if readiness >= 70 else
    ("B", "#FDE68A", "#F59E0B") if readiness >= 55 else
    ("C", "#FDBA74", "#EA580C") if readiness >= 40 else
    ("D", "#FCA5A5", "#DC2626")
)

st.markdown('<p class="section-header">Live Profile Overview</p>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("Study Hours", f"{study_hours}h")
with c2: st.metric("Attendance", f"{attendance:.0f}%")
with c3: st.metric("Exam Score", f"{exam_score:.0f}")
with c4: st.metric("Assignments", f"{assignments_completed}%")
with c5: st.metric("Readiness Grade", grade[0])

st.markdown('<p class="section-header">Performance Indicators</p>', unsafe_allow_html=True)

def render_bar(label, value, max_val, color):
    pct = min(int(value / max_val * 100), 100)
    st.markdown(f"""
    <div class="profile-bar-wrap">
      <div class="profile-bar-header"><span>{label}</span><span>{value} / {max_val}</span></div>
      <div class="profile-bar-bg">
        <div class="profile-bar-fill" style="width:{pct}%;background:{color};"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

render_bar("Exam Score",           exam_score,             100, "#818CF8")
render_bar("Previous Score",       previous_score,         100, "#6366F1")
render_bar("Attendance",           attendance,             100, "#22D3EE")
render_bar("Study Hours / Day",    study_hours,            10,  "#34D399")
render_bar("Assignments Done",     assignments_completed,  100, "#FBBF24")
render_bar("Sleep Hours",          sleep_hours,            8,   "#A78BFA")
render_bar("Internet (inverted)",  max(0, 12 - internet_usage), 12, "#F87171")

# ---------------------------------------------------
# RADAR CHART
# ---------------------------------------------------
st.markdown('<p class="section-header">Skill Radar</p>', unsafe_allow_html=True)

radar_categories = [
    "Exam Score", "Prev Score", "Attendance",
    "Study Hours", "Assignments", "Sleep", "Focus"
]
radar_values = [
    exam_score,
    previous_score,
    attendance,
    min(study_hours / 15 * 100, 100),
    assignments_completed,
    min(sleep_hours / 8 * 100, 100),
    max(0, 100 - internet_usage / 15 * 100),
]

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=radar_values + [radar_values[0]],
    theta=radar_categories + [radar_categories[0]],
    fill='toself',
    fillcolor='rgba(99,102,241,0.15)',
    line=dict(color='#818CF8', width=2),
    name='Student'
))
fig_radar.add_trace(go.Scatterpolar(
    r=[75, 75, 75, 75, 75, 75, 75, 75],
    theta=radar_categories + [radar_categories[0]],
    fill='toself',
    fillcolor='rgba(255,255,255,0.03)',
    line=dict(color='#334155', width=1, dash='dot'),
    name='Target (75)'
))
fig_radar.update_layout(
    polar=dict(
        bgcolor='#0F1523',
        radialaxis=dict(visible=True, range=[0, 100], color='#334155', gridcolor='#1E2A3D'),
        angularaxis=dict(color='#64748B')
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=True,
    legend=dict(font=dict(color='#94A3B8'), bgcolor='rgba(0,0,0,0)'),
    margin=dict(t=20, b=20, l=40, r=40),
    height=380
)
st.plotly_chart(fig_radar, use_container_width=True)

# ---------------------------------------------------
# BAR CHART
# ---------------------------------------------------
chart_data = pd.DataFrame({
    "Metric": ["Exam Score", "Prev Score", "Attendance", "Study×10", "Assignments", "Sleep×12"],
    "Value":  [exam_score, previous_score, attendance,
               study_hours * 10, assignments_completed, sleep_hours * 12],
    "Color":  ["#818CF8","#6366F1","#22D3EE","#34D399","#FBBF24","#A78BFA"]
})
fig_bar = px.bar(
    chart_data, x="Metric", y="Value",
    title="Analytics Overview (scaled to 100)",
    color="Color", color_discrete_map={c: c for c in chart_data["Color"]},
    text_auto=".0f"
)
fig_bar.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#94A3B8'),
    title_font=dict(color='#E2E8F0'),
    xaxis=dict(gridcolor='#1E2A3D'),
    yaxis=dict(gridcolor='#1E2A3D', range=[0, 110]),
    showlegend=False,
    margin=dict(t=50, b=20),
    height=340
)
st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------
st.markdown('<p class="section-header">Placement Prediction</p>', unsafe_allow_html=True)

predict_col, _ = st.columns([1, 2])
with predict_col:
    predict_clicked = st.button("🚀 Run Placement Prediction")

# ---------------------------------------------------
# PREDICTION OUTPUT
# ---------------------------------------------------
if predict_clicked:

    if not model_loaded:
        st.error("❌ Model file `student_placement_pipeline.pkl` not found. Please place it in the same directory as this app.")
    else:
        input_data = pd.DataFrame({
            "study_hours":            [study_hours],
            "attendance":             [attendance],
            "sleep_hours":            [sleep_hours],
            "internet_usage":         [internet_usage],
            "assignments_completed":  [assignments_completed],
            "previous_score":         [previous_score],
            "exam_score":             [exam_score],
        })

        prediction  = pipeline.predict(input_data)[0]
        try:
            probability = pipeline.predict_proba(input_data)[0][1] * 100
        except Exception:
            # Fallback: use readiness score as a calibrated estimate
            probability = readiness

        # --- Result box ---
        if prediction == 1:
            st.markdown(
                '<div class="result-placed">✅ &nbsp;LIKELY TO BE PLACED</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-notplaced">❌ &nbsp;PLACEMENT AT RISK</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        # --- Gauge + Readiness side-by-side ---
        g1, g2 = st.columns(2)

        with g1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(probability, 1),
                delta={'reference': 60, 'increasing': {'color': "#6EE7B7"}, 'decreasing': {'color': "#FCA5A5"}},
                title={'text': "Model Placement Probability", 'font': {'color': '#94A3B8', 'size': 14}},
                number={'suffix': "%", 'font': {'color': '#E2E8F0', 'size': 32}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#334155'},
                    'bar': {'color': "#6366F1", 'thickness': 0.25},
                    'bgcolor': '#0F1523',
                    'bordercolor': '#1E2A3D',
                    'steps': [
                        {'range': [0, 40],  'color': '#2D0A0A'},
                        {'range': [40, 65], 'color': '#1C1500'},
                        {'range': [65, 100],'color': '#022C22'},
                    ],
                    'threshold': {
                        'line': {'color': '#F8FAFC', 'width': 2},
                        'thickness': 0.75,
                        'value': probability
                    }
                }
            ))
            gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94A3B8'),
                height=280,
                margin=dict(t=40, b=10, l=20, r=20)
            )
            st.plotly_chart(gauge, use_container_width=True)

        with g2:
            readiness_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=readiness,
                title={'text': "Weighted Readiness Score", 'font': {'color': '#94A3B8', 'size': 14}},
                number={'suffix': "/100", 'font': {'color': '#E2E8F0', 'size': 32}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#334155'},
                    'bar': {'color': "#22D3EE", 'thickness': 0.25},
                    'bgcolor': '#0F1523',
                    'bordercolor': '#1E2A3D',
                    'steps': [
                        {'range': [0, 40],  'color': '#2D0A0A'},
                        {'range': [40, 65], 'color': '#1C1500'},
                        {'range': [65, 100],'color': '#022C22'},
                    ],
                }
            ))
            readiness_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94A3B8'),
                height=280,
                margin=dict(t=40, b=10, l=20, r=20)
            )
            st.plotly_chart(readiness_gauge, use_container_width=True)

        # --- Factor contribution breakdown ---
        st.markdown('<p class="section-header">Factor Contribution Analysis</p>', unsafe_allow_html=True)

        factors = pd.DataFrame({
            "Factor": ["Exam Score", "Prev Score", "Attendance", "Study Hours", "Assignments", "Sleep", "Internet Focus"],
            "Weight (%)": [28, 20, 18, 14, 10, 6, 4],
            "Your Score": [
                exam_score,
                previous_score,
                attendance,
                min(study_hours / 10 * 100, 100),
                assignments_completed,
                min(sleep_hours / 8 * 100, 100),
                max(0, 100 - internet_usage / 12 * 100)
            ]
        })
        factors["Contribution"] = (factors["Weight (%)"] / 100) * factors["Your Score"]
        factors["Max Possible"] = factors["Weight (%)"]
        factors["Gap"] = factors["Max Possible"] - factors["Contribution"]

        fig_factor = go.Figure()
        fig_factor.add_trace(go.Bar(
            name='Your Contribution',
            x=factors["Factor"],
            y=factors["Contribution"].round(1),
            marker_color='#6366F1',
            text=factors["Contribution"].round(1),
            textposition='auto'
        ))
        fig_factor.add_trace(go.Bar(
            name='Gap to Max',
            x=factors["Factor"],
            y=factors["Gap"].round(1),
            marker_color='#1E2A3D',
            text=factors["Gap"].round(1),
            textposition='auto'
        ))
        fig_factor.update_layout(
            barmode='stack',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94A3B8'),
            legend=dict(font=dict(color='#94A3B8'), bgcolor='rgba(0,0,0,0)'),
            xaxis=dict(gridcolor='#1E2A3D'),
            yaxis=dict(gridcolor='#1E2A3D', title="Points (out of total weight)"),
            margin=dict(t=20, b=20),
            height=320,
        )
        st.plotly_chart(fig_factor, use_container_width=True)

        # ---------------------------------------------------
        # AI RECOMMENDATIONS
        # ---------------------------------------------------
        st.markdown('<p class="section-header">AI Recommendations</p>', unsafe_allow_html=True)

        recs = get_smart_recommendations(
            study_hours, attendance, sleep_hours,
            internet_usage, assignments_completed,
            previous_score, exam_score, probability
        )

        if not recs:
            st.markdown("""
            <div class="rec-tip">
              <div class="rec-title">🏆 Outstanding Profile</div>
              All metrics are in excellent shape. Your focus should now shift entirely to
              interview preparation: DSA, system design, mock interviews, and company research.
              You're placement-ready.
            </div>
            """, unsafe_allow_html=True)
        else:
            for level, title, msg, impact in recs:
                badge = {"critical": "🔴 CRITICAL", "warning": "🟡 IMPROVE", "tip": "🟢 TIP"}[level]
                css_class = f"rec-{level}"
                st.markdown(f"""
                <div class="{css_class}">
                  <div class="rec-title">{badge} · {impact} &nbsp;|&nbsp; {title}</div>
                  {msg}
                </div>
                """, unsafe_allow_html=True)

        # --- Priority Action Plan ---
        st.markdown('<p class="section-header">Priority Action Plan</p>', unsafe_allow_html=True)

        priority_items = []
        if exam_score < 65:
            priority_items.append(("1", "Exam Prep", "Schedule daily mock tests and topic revision for 2+ hrs", "#EF4444"))
        if attendance < 75:
            priority_items.append(("2", "Attendance", "Attend every class this week without exception", "#EF4444"))
        if study_hours < 4:
            priority_items.append(("3", "Study Time", "Add 1 focused hour to your daily schedule this week", "#F59E0B"))
        if assignments_completed < 60:
            priority_items.append(("4", "Assignments", "Clear your assignment backlog — complete 10 this week", "#F59E0B"))
        if internet_usage > 6:
            priority_items.append(("5", "Screen Time", "Enable app limits: max 3 hrs non-study internet/day", "#10B981"))
        if sleep_hours < 6.5:
            priority_items.append(("6", "Sleep", "Move bedtime 30 min earlier starting tonight", "#10B981"))

        if not priority_items:
            st.success("✅ No critical actions required. Maintain your current habits and focus on interview prep.")
        else:
            cols = st.columns(min(len(priority_items), 3))
            for i, (num, area, action, color) in enumerate(priority_items[:3]):
                with cols[i]:
                    st.markdown(f"""
                    <div style="background:#0F1523;border:1px solid #1E2A3D;border-top:3px solid {color};
                                border-radius:12px;padding:18px 16px;height:100%;">
                        <div style="font-family:'Space Grotesk',sans-serif;font-size:0.7rem;
                                    text-transform:uppercase;letter-spacing:0.1em;color:{color};
                                    margin-bottom:6px;font-weight:700;">Priority {num} · {area}</div>
                        <div style="font-size:0.88rem;color:#CBD5E1;line-height:1.5;">{action}</div>
                    </div>
                    """, unsafe_allow_html=True)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.markdown(
    '<div class="footer">Built with Streamlit & Scikit-learn &nbsp;·&nbsp; '
    'Readiness score is computed independently of the ML model using domain-weighted metrics</div>',
    unsafe_allow_html=True
)
