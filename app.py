import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("car_price_predictor_rf.pkl")

st.title("🚗 Car Price Prediction Dashboard")

# User Inputs
car_name = st.selectbox("Car Brand/Name", ['Toyota','Honda','BMW','Audi','Hyundai'])
present_price = st.number_input("Present Price (in $)", min_value=500, max_value=100000, value=10000)
driven_kms = st.number_input("Driven KMs", min_value=0, max_value=300000, value=50000)
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
selling_type = st.selectbox("Selling Type", ['First Owner','Second Owner','Third Owner'])
transmission = st.selectbox("Transmission", ['Manual','Automatic'])
owner = st.selectbox("Number of Owners", ['First','Second','Third'])
year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, value=2015)

# Calculate Car Age
import datetime
current_year = datetime.datetime.now().year
car_age = current_year - year

# Optional price difference feature
price_diff = present_price - present_price  # Could be zero for input

# Encode categorical features same as training
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# This should match the training encoding, for demo we will map manually
fuel_map = {'Petrol':0, 'Diesel':1, 'CNG':2}
trans_map = {'Manual':0, 'Automatic':1}
owner_map = {'First':0, 'Second':1, 'Third':2}
car_name_map = {'Toyota':0,'Honda':1,'BMW':2,'Audi':3,'Hyundai':4}
selling_type_map = {'First Owner':0,'Second Owner':1,'Third Owner':2}

# Prepare input DataFrame
input_df = pd.DataFrame({
    'car_name':[car_name_map[car_name]],
    'present_price':[present_price],
    'driven_kms':[driven_kms],
    'fuel_type':[fuel_map[fuel_type]],
    'selling_type':[selling_type_map[selling_type]],
    'transmission':[trans_map[transmission]],
    'owner':[owner_map[owner]],
    'car_age':[car_age],
    'price_diff':[price_diff]
})

# Predict Button
if st.button("Predict Selling Price"):
    price_pred = model.predict(input_df)
    st.success(f"💰 Predicted Selling Price: ${price_pred[0]:.2f}")
