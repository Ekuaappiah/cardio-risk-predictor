import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Load Model ---
@st.cache_resource
def load_model():
    model_path = Path('models/heart_disease_predictor.pkl')
    model = joblib.load(model_path)
    print("Model features:", model.feature_names_in_)  # Debug: Check expected features
    return model


model = load_model()


# --- UI Design ---
def main():
    st.markdown("<h1 style='text-align: center; color: #d63384;'>❤️ Heart Disease Risk Assessment</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Enter patient health metrics to assess cardiovascular risk</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    # Information callout
    with st.expander("ℹ️ About This Tool", expanded=False):
        st.markdown("""
           ### How to use this tool
           1. Enter all patient health metrics in the form below
           2. Click the "Predict Risk" button to evaluate cardiovascular risk
           3. Review the assessment results and recommendations

           ### Disclaimer
           This tool is intended for clinical decision support only and should not replace professional medical judgment.
           Always consult with a healthcare provider before making medical decisions.
           """)

    with st.form("prediction_form"):
        st.markdown("### 🩺 Patient Details")
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            age = st.slider("Age", 20, 100, 50)
            sex = st.radio("Sex", ["Male", "Female"])
            cp = st.selectbox("Chest Pain Type", [
                "Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"
            ])
            trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 120)

        with col2:
            chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
            fbs = st.checkbox("Fasting BS > 120 mg/dl")
            restecg = st.selectbox("Resting ECG", [
                "Normal", "ST-T abnormality", "LV hypertrophy"
            ])
            thalach = st.number_input("Max Heart Rate", 60, 220, 150)

        with col3:
            exang = st.checkbox("Exercise Angina")
            oldpeak = st.number_input("ST Depression", 0.0, 6.2, 1.0, step=0.1)
            slope = st.selectbox("Slope", [
                "Upsloping", "Flat", "Downsloping"
            ])
            ca = st.slider("Major Vessels", 0, 3, 0)

        submitted = st.form_submit_button("🔍 Predict Risk")

        if submitted:
            # Encode all categorical variables
            feature_dict = {
                'age': age,
                'sex': 1 if sex == "Male" else 0,
                'cp': ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"].index(cp),
                'trestbps': trestbps,
                'chol': chol,
                'fbs': int(fbs),
                'restecg': ["Normal", "ST-T abnormality", "LV hypertrophy"].index(restecg),
                'thalach': thalach,
                'exang': int(exang),
                'oldpeak': oldpeak,
                'slope': ["Upsloping", "Flat", "Downsloping"].index(slope),
                'ca': ca,
                'thal': 2  # Default value for thalassemia
            }

            # Create DataFrame with ALL features in correct order
            features = pd.DataFrame([feature_dict], columns=model.feature_names_in_)

            try:
                # Make prediction
                probability = model.predict_proba(features)[0][1]
                display_results(probability)

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.warning("Please ensure all required features are provided")


PRIMARY_COLOR = "#d63384" 
def create_gauge_chart(probability):
    # Create gauge chart for risk visualization
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Level", 'font': {'size': 24, 'color': PRIMARY_COLOR}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "gray"},
            'bar': {'color': PRIMARY_COLOR},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#e8f5e9'},
                {'range': [30, 60], 'color': '#fff3e0'},
                {'range': [60, 100], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)


def display_results(probability):
    st.markdown("### 🧠 Clinical Risk Assessment")

    create_gauge_chart(probability)

    LOW_RISK_THRESHOLD = 0.3
    MEDIUM_RISK_THRESHOLD = 0.6

    if probability <= LOW_RISK_THRESHOLD:
        recommendations = [
            "Continue routine health screenings",
            "Maintain healthy lifestyle habits",
            "Schedule next check-up in 12 months"
        ]
    elif probability <= MEDIUM_RISK_THRESHOLD:
        recommendations = [
            "Schedule follow-up clinical evaluation",
            "Consider lifestyle modifications",
            "Monitor blood pressure and cholesterol",
            "Follow-up within 3–6 months"
        ]
    else:
        recommendations = [
            "Urgent physician consultation recommended",
            "Consider cardiac stress test and imaging",
            "Evaluate medication therapy options",
            "Follow-up within 2–4 weeks"
        ]
        show_emergency_protocols()

    st.markdown("### 📋 Recommendations")
    for rec in recommendations:
        st.markdown(f"- {rec}")


def show_result(title, probability, emoji, recommendation, box_type):
    with st.container():
        col1, col2 = st.columns([1, 10])
        with col1:
            st.markdown(f"<h1 style='text-align: center;'>{emoji}</h1>", unsafe_allow_html=True)
        with col2:
            if box_type == "success":
                st.success(f"**{title}**\n\nProbability: {probability:.0%}\n\n*Recommendation:* {recommendation}")
            elif box_type == "warning":
                st.warning(f"**{title}**\n\nProbability: {probability:.0%}\n\n*Recommendation:* {recommendation}")
            else:
                st.error(f"**{title}**\n\nProbability: {probability:.0%}\n\n*Recommendation:* {recommendation}")


def show_emergency_protocols():
    st.markdown("""
    <div style=" border-left:6px solid #dc3545; padding:16px 20px; border-radius:8px; margin-top:30px; 
    margin-bottom:30px;">
        <h4 style="margin-top:0; color:#dc3545;">⚠️ Emergency Protocols</h4>
        <ul style="padding-left: 1.2em; margin: 0;">
            <li style="margin-bottom: 6px;">Immediate ECG and stress testing recommended</li>
            <li style="margin-bottom: 6px;">Consider initiating lipid-lowering therapy</li>
            <li>Monitor blood pressure twice daily</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)



# --- Custom CSS ---
st.markdown("""
<style>
    .stButton>button {
        background-color: #d63384;
        color: white;
        font-weight: bold;
        padding: 0.6em 1.2em;
        border-radius: 8px;
        margin-top: 1.5em;
    }
    .stSlider > div {
        padding: 5px 0;
    }
    .stNumberInput, .stSelectbox, .stRadio {
        margin-bottom: 1em;
    }
    .stMarkdown h3 {
        color: #d63384;
    }
</style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()