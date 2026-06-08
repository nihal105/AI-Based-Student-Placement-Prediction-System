import streamlit as st
import pandas as pd
import joblib

# Load the saved pipeline
pipeline = joblib.load("student_placement_pipeline.pkl")

# App title
st.title("Student Placement Prediction System")

st.write("Enter student details below:")

# Input fields
study_hours = st.number_input("Study Hours", min_value=0.0)
attendance = st.number_input("Attendance", min_value=0.0)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0)
internet_usage = st.number_input("Internet Usage", min_value=0.0)
assignments_completed = st.number_input("Assignments Completed", min_value=0)
previous_score = st.number_input("Previous Score", min_value=0.0)
exam_score = st.number_input("Exam Score", min_value=0.0)

# Prediction button
if st.button("Predict Placement Status"):

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

    result = "Placed" if prediction[0] == 1 else "Not Placed"

    st.success(f"Placement Prediction: {result}")