import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Prediction")
st.write("Predict a student's final grade using academic and social factors.")

st.sidebar.header("Student Information")

school = st.sidebar.selectbox("School", ["GP", "MS"])
sex = st.sidebar.selectbox("Sex", ["F", "M"])
age = st.sidebar.slider("Age", 15, 22, 17)
address = st.sidebar.selectbox("Address", ["U", "R"])
famsize = st.sidebar.selectbox("Family Size", ["LE3", "GT3"])
Pstatus = st.sidebar.selectbox("Parent Status", ["T", "A"])

Medu = st.sidebar.slider("Mother Education", 0, 4, 2)
Fedu = st.sidebar.slider("Father Education", 0, 4, 2)
traveltime = st.sidebar.slider("Travel Time", 1, 4, 1)
studytime = st.sidebar.slider("Study Time", 1, 4, 2)
failures = st.sidebar.slider("Past Class Failures", 0, 4, 0)

schoolsup = st.sidebar.selectbox("School Support", ["yes", "no"])
famsup = st.sidebar.selectbox("Family Support", ["yes", "no"])
paid = st.sidebar.selectbox("Paid Classes", ["yes", "no"])
activities = st.sidebar.selectbox("Activities", ["yes", "no"])
internet = st.sidebar.selectbox("Internet Access", ["yes", "no"])
romantic = st.sidebar.selectbox("Romantic Relationship", ["yes", "no"])

famrel = st.sidebar.slider("Family Relationship", 1, 5, 4)
freetime = st.sidebar.slider("Free Time", 1, 5, 3)
goout = st.sidebar.slider("Going Out", 1, 5, 3)
Dalc = st.sidebar.slider("Workday Alcohol Consumption", 1, 5, 1)
Walc = st.sidebar.slider("Weekend Alcohol Consumption", 1, 5, 1)
health = st.sidebar.slider("Health", 1, 5, 3)
absences = st.sidebar.slider("Absences", 0, 100, 5)

G1 = st.sidebar.slider("First Period Grade", 0, 20, 10)
G2 = st.sidebar.slider("Second Period Grade", 0, 20, 10)

input_data = pd.DataFrame([{
    "school": school,
    "sex": sex,
    "age": age,
    "address": address,
    "famsize": famsize,
    "Pstatus": Pstatus,
    "Medu": Medu,
    "Fedu": Fedu,
    "Mjob": "other",
    "Fjob": "other",
    "reason": "course",
    "guardian": "mother",
    "traveltime": traveltime,
    "studytime": studytime,
    "failures": failures,
    "schoolsup": schoolsup,
    "famsup": famsup,
    "paid": paid,
    "activities": activities,
    "nursery": "yes",
    "higher": "yes",
    "internet": internet,
    "romantic": romantic,
    "famrel": famrel,
    "freetime": freetime,
    "goout": goout,
    "Dalc": Dalc,
    "Walc": Walc,
    "health": health,
    "absences": absences,
    "G1": G1,
    "G2": G2
}])

if st.button("Predict Final Grade"):
    prediction = model.predict(input_data)[0]
    prediction = max(0, min(20, prediction))

    st.subheader("Predicted Final Grade")
    st.success(f"{prediction:.2f} / 20")