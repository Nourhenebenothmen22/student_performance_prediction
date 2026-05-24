import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }

    .hero {
        padding: 35px;
        border-radius: 20px;
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 10px;
    }

    .hero p {
        font-size: 18px;
        color: #e6e6e6;
    }

    .section-card {
        padding: 25px;
        border-radius: 18px;
        background-color: white;
        box-shadow: 0px 5px 18px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    .prediction-card {
        padding: 30px;
        border-radius: 20px;
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        text-align: center;
        box-shadow: 0px 8px 25px rgba(0,0,0,0.18);
    }

    .prediction-number {
        font-size: 55px;
        font-weight: bold;
    }

    .small-text {
        font-size: 14px;
        color: #555;
    }

    .stButton > button {
        width: 100%;
        height: 52px;
        border-radius: 14px;
        font-size: 18px;
        font-weight: bold;
        background-color: #1e3c72;
        color: white;
        border: none;
    }

    .stButton > button:hover {
        background-color: #2a5298;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Hero section
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>🎓 Student Performance Prediction</h1>
    <p>Machine Learning app that predicts a student's final grade based on academic, social, and personal factors.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Layout
# -----------------------------
left_col, right_col = st.columns([2, 1])

with left_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📌 Project Overview")

    st.write("""
    This application uses machine learning to predict a student's final academic grade.
    The model was trained using student demographic, educational, and behavioral data.
    """)

    st.markdown("""
    **Tech Stack:** Python, Pandas, Scikit-learn, Streamlit  
    **Models:** Linear Regression, Random Forest Regressor  
    **Prediction Target:** Final Grade `G3`
    """)

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("⚙️ Model Info")

    st.metric("Grade Scale", "0 - 20")
    st.metric("Input Features", "30+")
    st.metric("Model Type", "Regression")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Input form
# -----------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("🧾 Enter Student Information")

with st.form("prediction_form"):

    tab1, tab2, tab3 = st.tabs([
        "👤 Personal Info",
        "🏫 Academic Info",
        "🌐 Social & Lifestyle"
    ])

    with tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            school = st.selectbox("School", ["GP", "MS"])
            sex = st.selectbox("Sex", ["F", "M"])
            age = st.slider("Age", 15, 22, 17)

        with col2:
            address = st.selectbox("Address", ["U", "R"])
            famsize = st.selectbox("Family Size", ["LE3", "GT3"])
            Pstatus = st.selectbox("Parent Status", ["T", "A"])

        with col3:
            Medu = st.slider("Mother Education", 0, 4, 2)
            Fedu = st.slider("Father Education", 0, 4, 2)
            guardian = st.selectbox("Guardian", ["mother", "father", "other"])

    with tab2:
        col1, col2, col3 = st.columns(3)

        with col1:
            studytime = st.slider("Study Time", 1, 4, 2)
            failures = st.slider("Past Class Failures", 0, 4, 0)
            absences = st.slider("Absences", 0, 100, 5)

        with col2:
            G1 = st.slider("First Period Grade", 0, 20, 10)
            G2 = st.slider("Second Period Grade", 0, 20, 10)
            traveltime = st.slider("Travel Time", 1, 4, 1)

        with col3:
            schoolsup = st.selectbox("School Support", ["yes", "no"])
            famsup = st.selectbox("Family Support", ["yes", "no"])
            paid = st.selectbox("Paid Classes", ["yes", "no"])

    with tab3:
        col1, col2, col3 = st.columns(3)

        with col1:
            activities = st.selectbox("Extra Activities", ["yes", "no"])
            internet = st.selectbox("Internet Access", ["yes", "no"])
            romantic = st.selectbox("Romantic Relationship", ["yes", "no"])

        with col2:
            famrel = st.slider("Family Relationship", 1, 5, 4)
            freetime = st.slider("Free Time", 1, 5, 3)
            goout = st.slider("Going Out", 1, 5, 3)

        with col3:
            Dalc = st.slider("Workday Alcohol Consumption", 1, 5, 1)
            Walc = st.slider("Weekend Alcohol Consumption", 1, 5, 1)
            health = st.slider("Health", 1, 5, 3)

    submitted = st.form_submit_button("Predict Final Grade")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Prepare input data
# -----------------------------
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
    "guardian": guardian,
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

# -----------------------------
# Prediction result
# -----------------------------
if submitted:
    prediction = model.predict(input_data)[0]
    prediction = max(0, min(20, prediction))

    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    st.markdown("<h2>🎯 Predicted Final Grade</h2>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="prediction-number">{prediction:.2f} / 20</div>',
        unsafe_allow_html=True
    )

    if prediction >= 15:
        st.markdown("<h3>Excellent performance expected 🚀</h3>", unsafe_allow_html=True)
    elif prediction >= 10:
        st.markdown("<h3>Average performance expected 👍</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3>Student may need academic support ⚠️</h3>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.progress(int((prediction / 20) * 100))

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("""
<p style="text-align:center; color:gray;">
Built with Python, Scikit-learn and Streamlit | Student Performance Prediction Project
</p>
""", unsafe_allow_html=True)