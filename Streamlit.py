import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# Load Model, Scaler, Selector
# =========================
model = joblib.load('coffee_sales_model.pkl')
scaler = joblib.load('scaler.pkl')
selector = joblib.load('selector.pkl')

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Coffee Shop Revenue Predictor",
    page_icon="☕",
    layout="centered"
)

# =========================
# Custom Modern CSS
# =========================
st.markdown(
    """
    <style>
    /* Background Gradient */
    .stApp {
        background: linear-gradient(to right, #f8f9fa, #e3f2fd);
        font-family: 'Segoe UI', sans-serif;
    }

    /* Title Styling */
    .title {
        text-align: center;
        color: #2e4053;
        font-size: 38px;
        font-weight: bold;
        padding-bottom: 10px;
    }

    /* Form Card */
    .stForm {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }

    /* Button */
    div.stButton > button {
        background: linear-gradient(90deg, #6a11cb, #2575fc);
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 0.6em 1.5em;
        border: none;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #2575fc, #6a11cb);
        color: #fff;
        transform: scale(1.05);
        transition: 0.3s;
    }

    /* Prediction Card */
    .prediction-card {
        background: #ffffff;
        border-left: 8px solid #2575fc;
        padding: 20px;
        margin-top: 25px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
        text-align: center;
    }
    .prediction-card h3 {
        font-size: 26px;
        color: #2e4053;
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Title
# =========================
st.markdown("<h1 class='title'>☕ Coffee Shop Revenue Predictor</h1>", unsafe_allow_html=True)
st.markdown("### Fill in your daily shop details and get a modern revenue prediction instantly!")

# =========================
# Form Section
# =========================
with st.form("prediction_form"):
    st.markdown("#### 📋 Enter Daily Shop Details")

    col1, col2 = st.columns(2)

    with col1:
        day_of_week = st.slider("Day of Week (1=Mon, 7=Sun)", 1, 7, 4)
        is_weekend = st.selectbox("Is Weekend?", [0, 1])
        month = st.slider("Month (1-12)", 1, 12, 6)
        temperature = st.number_input("Temperature (°C)", value=25.0)
        is_raining = st.selectbox("Is Raining?", [0, 1])
        rainfall = st.number_input("Rainfall (mm)", value=0.0)
        is_holiday = st.selectbox("Is Holiday?", [0, 1])
        promotion_active = st.selectbox("Promotion Active?", [0, 1])
        nearby_events = st.selectbox("Nearby Events?", [0, 1])

    with col2:
        staff_count = st.slider("Staff Count", 1, 10, 5)
        machine_issues = st.selectbox("Machine Issues?", [0, 1])
        num_customers = st.number_input("Number of Customers", value=50)
        coffee_sales = st.number_input("Coffee Sales", value=60)
        pastry_sales = st.number_input("Pastry Sales", value=30)
        sandwich_sales = st.number_input("Sandwich Sales", value=20)
        customer_satisfaction = st.slider("Customer Satisfaction (1-10)", 1.0, 10.0, 8.0)
        day_of_year = st.slider("Day of Year", 1, 366, 180)
        week_of_year = st.slider("Week of Year", 1, 52, 26)
        quarter = st.slider("Quarter (1-4)", 1, 4, 2)
        day_name_encoded = st.slider("Day Name Encoded", 0, 6, 3)
        season_encoded = st.slider("Season Encoded", 0, 3, 2)

    submitted = st.form_submit_button("🚀 Predict Revenue")

# =========================
# Prediction Logic
# =========================
if submitted:
    input_data = {
        'Day_of_Week': day_of_week,
        'Is_Weekend': is_weekend,
        'Month': month,
        'Temperature_C': temperature,
        'Is_Raining': is_raining,
        'Rainfall_mm': rainfall,
        'Is_Holiday': is_holiday,
        'Promotion_Active': promotion_active,
        'Nearby_Events': nearby_events,
        'Staff_Count': staff_count,
        'Machine_Issues': machine_issues,
        'Num_Customers': num_customers,
        'Coffee_Sales': coffee_sales,
        'Pastry_Sales': pastry_sales,
        'Sandwich_Sales': sandwich_sales,
        'Customer_Satisfaction': customer_satisfaction,
        'Day_of_Year': day_of_year,
        'Week_of_Year': week_of_year,
        'Quarter': quarter,
        'Day_Name_Encoded': day_name_encoded,
        'Season_Encoded': season_encoded
    }

    input_df = pd.DataFrame([input_data])

    # Preprocess
    scaled_input = scaler.transform(input_df)
    selected_input = selector.transform(scaled_input)

    # Predict
    prediction = model.predict(selected_input)[0]

    # Display Result
    st.markdown(
        f"""
        <div class="prediction-card">
            <h3>💰 Predicted Daily Revenue:</h3>
            <h2 style="color:#2575fc;">${prediction:,.2f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
