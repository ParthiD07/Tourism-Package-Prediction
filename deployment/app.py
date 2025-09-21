import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ================================
# App Title & Description
# ================================
st.set_page_config(page_title="Tourism Package Prediction", page_icon="🌍", layout="centered")

st.title("🌍 Tourism Package Prediction App")
st.markdown(
    """
    Provide customer details below to predict whether they are likely to 
    **opt for a tourism package**.
    """
)

# ================================
# Load Model from Hugging Face Hub
# ================================
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Parthi07/Package-Prediction-Model",
        filename="models/best_package_prediction_model_v1.joblib"
    )
    return joblib.load(model_path)

model = load_model()

city_tier_map = {"Tier 1": 1, "Tier 2": 2, "Tier 3": 3}

# ================================
# Tabs for Input Sections
# ================================
tabs = st.tabs([
    "👤 Personal Information", 
    "💰 Lifestyle & Financial", 
    "✈️ Travel Preferences", 
    "👨‍👩‍👧 Family & Trips", 
    "📞 Sales Interaction"
])

with tabs[0]:
    age = st.number_input("Age of Customer", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Female", "Male"])
    marital_status = st.selectbox("Marital Status", ["Single", "Divorced", "Married", "Unmarried"])
    occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
    city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])

with tabs[1]:
    monthly_income = st.number_input("Monthly Income", min_value=100, max_value=200000, value=10000)
    own_car = st.radio("Owns a Car?", ["Yes", "No"], horizontal=True)
    passport = st.radio("Has Passport?", ["Yes", "No"], horizontal=True)

with tabs[2]:
    product_pitched = st.selectbox("Product Pitched", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])
    preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])

with tabs[3]:
    num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=5, value=1)
    num_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=3, value=0)
    num_trips = st.number_input("Number of Trips", min_value=1, max_value=22, value=3)

with tabs[4]:
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    duration_of_pitch = st.number_input("Pitch Duration (minutes)", min_value=0, max_value=150, value=30)
    num_followups = st.number_input("Number of Followups", min_value=1, max_value=6, value=1)
    pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)

# ================================
# Prepare Input Data
# ================================
input_data = pd.DataFrame([{
    "TypeofContact": type_of_contact,
    "CityTier": city_tier_map[city_tier],
    "Occupation": occupation,
    "Gender": gender,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "Designation": designation,
    "NumberOfPersonVisiting": num_person_visiting,
    "NumberOfFollowups": num_followups,
    "NumberOfTrips": num_trips,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "NumberOfChildrenVisiting": num_children_visiting,
    "MonthlyIncome": monthly_income,
    "DurationOfPitch": duration_of_pitch,
    "Age": age,
    "Passport": 1 if passport == "Yes" else 0,
    "OwnCar": 1 if own_car == "Yes" else 0
}])

# ================================
# Prediction
# ================================
CLASSIFICATION_THRESHOLD = 0.45

if st.button("🔮 Predict", use_container_width=True):
    proba = float(model.predict_proba(input_data)[0][1])
    prediction = 1 if proba >= CLASSIFICATION_THRESHOLD else 0

    result = "✅ Package Opted" if prediction == 1 else "❌ Package Not Opted"
    confidence = f"{proba * 100:.2f}"

    st.markdown("---")
    st.subheader("📊 Prediction Result")
    st.success(f"**{result}** with {confidence}% confidence")

    st.write("### Entered Customer Profile:")
    st.dataframe(input_data.T, use_container_width=True)
