import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(90deg,#4F46E5,#7C3AED);
    color: white;
    border-radius: 12px;
    height: 3.2em;
    font-size: 18px;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.02);
    background: linear-gradient(90deg,#7C3AED,#4F46E5);
}

.metric-card {
    background-color: #161B22;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 0px 10px rgba(255,255,255,0.05);
}

.title {
    text-align:center;
    font-size:48px;
    font-weight:bold;
    color:#7C3AED;
}

.subtitle {
    text-align:center;
    font-size:20px;
    color:lightgray;
    margin-bottom:30px;
}

.prediction-box {
    padding:25px;
    border-radius:15px;
    text-align:center;
    font-size:30px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
pipeline = joblib.load("student_placement_pipeline.pkl")

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown('<p class="title">🎓 Student Placement Prediction System</p>', unsafe_allow_html=True)

st.markdown(
    '<p class="subtitle">AI Powered Placement Prediction Dashboard</p>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("📌 Student Inputs")

study_hours = st.sidebar.slider("📚 Study Hours", 0.0, 15.0, 5.0)
attendance = st.sidebar.slider("🏫 Attendance (%)", 0.0, 100.0, 75.0)
sleep_hours = st.sidebar.slider("😴 Sleep Hours", 0.0, 12.0, 7.0)
internet_usage = st.sidebar.slider("🌐 Internet Usage (hrs)", 0.0, 15.0, 4.0)
assignments_completed = st.sidebar.slider("📝 Assignments Completed", 0, 100, 50)
previous_score = st.sidebar.slider("📊 Previous Score", 0.0, 100.0, 60.0)
exam_score = st.sidebar.slider("🎯 Exam Score", 0.0, 100.0, 70.0)

# ---------------------------------------------------
# DISPLAY METRICS
# ---------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Study Hours", f"{study_hours} hrs")

with col2:
    st.metric("Attendance", f"{attendance}%")

with col3:
    st.metric("Exam Score", exam_score)

with col4:
    st.metric("Assignments", assignments_completed)

st.markdown("---")

# ---------------------------------------------------
# PROGRESS INDICATORS
# ---------------------------------------------------
st.subheader("📈 Student Performance Indicators")

st.write("Study Performance")
st.progress(min(int(study_hours * 6), 100))

st.write("Attendance Level")
st.progress(int(attendance))

st.write("Exam Readiness")
st.progress(int(exam_score))

# ---------------------------------------------------
# CHARTS
# ---------------------------------------------------
chart_data = pd.DataFrame({
    "Feature": [
        "Study Hours",
        "Attendance",
        "Sleep Hours",
        "Internet Usage",
        "Assignments",
        "Previous Score",
        "Exam Score"
    ],
    "Value": [
        study_hours,
        attendance,
        sleep_hours,
        internet_usage,
        assignments_completed,
        previous_score,
        exam_score
    ]
})

fig = px.bar(
    chart_data,
    x="Feature",
    y="Value",
    title="📊 Student Analytics",
    text_auto=True
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
if st.button("🚀 Predict Placement Status"):

    input_data = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance": [attendance],
        "sleep_hours": [sleep_hours],
        "internet_usage": [internet_usage],
        "assignments_completed": [assignments_completed],
        "previous_score": [previous_score],
        "exam_score": [exam_score]
    })

    prediction = pipeline.predict(input_data)

    # Probability
    try:
        probability = pipeline.predict_proba(input_data)[0][1] * 100
    except:
        probability = 0

    st.markdown("---")

    if prediction[0] == 1:

        st.markdown(
            f"""
            <div class="prediction-box" style="background:#064E3B;color:#6EE7B7;">
                ✅ STUDENT WILL BE PLACED
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="prediction-box" style="background:#7F1D1D;color:#FCA5A5;">
                ❌ STUDENT MAY NOT BE PLACED
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("## 🎯 Placement Probability")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        title={'text': "Placement Chance"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, 50], 'color': "#7F1D1D"},
                {'range': [50, 75], 'color': "#92400E"},
                {'range': [75, 100], 'color': "#064E3B"}
            ],
        }
    ))

    st.plotly_chart(gauge, use_container_width=True)

    # Recommendations
    st.markdown("## 💡 AI Recommendations")

    recommendations = []

    if attendance < 75:
        recommendations.append("✔ Improve attendance percentage.")

    if study_hours < 4:
        recommendations.append("✔ Increase daily study hours.")

    if exam_score < 60:
        recommendations.append("✔ Focus more on exam preparation.")

    if assignments_completed < 40:
        recommendations.append("✔ Complete more assignments.")

    if sleep_hours < 6:
        recommendations.append("✔ Maintain healthy sleep schedule.")

    if len(recommendations) == 0:
        st.success("Excellent performance! Keep it up 🚀")

    else:
        for rec in recommendations:
            st.warning(rec)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")

st.markdown(
    """
    <center>
        <h4>Made with ❤️ using Streamlit & Machine Learning</h4>
    </center>
    """,
    unsafe_allow_html=True
)