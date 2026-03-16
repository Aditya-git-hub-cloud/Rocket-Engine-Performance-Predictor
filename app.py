import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("rocket_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

st.title("🚀 Rocket Specific Impulse Predictor")

st.write("Enter rocket engine parameters to predict Specific Impulse")

# INPUTS

fuel_type = st.selectbox("Fuel Type", ["LH2", "RP-1", "CH4"])
oxidizer_type = st.selectbox("Oxidizer Type", ["LOX", "N2O4"])

chamber_pressure = st.number_input("Chamber Pressure", value=70)
oxidizer_fuel_ratio = st.number_input("Oxidizer Fuel Ratio", value=6)
combustion_temperature = st.number_input("Combustion Temperature", value=3500)
heat_capacity_ratio = st.number_input("Heat Capacity Ratio", value=1.2)
nozzle_expansion_ratio = st.number_input("Nozzle Expansion Ratio", value=40)
ambient_pressure = st.number_input("Ambient Pressure", value=1)

# Create dataframe
input_dict = {
    "fuel_type": fuel_type,
    "oxidizer_type": oxidizer_type,
    "chamber_pressure": chamber_pressure,
    "oxidizer_fuel_ratio": oxidizer_fuel_ratio,
    "combustion_temperature": combustion_temperature,
    "heat_capacity_ratio": heat_capacity_ratio,
    "nozzle_expansion_ratio": nozzle_expansion_ratio,
    "ambient_pressure": ambient_pressure
}

input_df = pd.DataFrame([input_dict])

# One-hot encoding
input_df = pd.get_dummies(input_df)

# Add missing columns
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[model_columns]

# Prediction
if st.button("Predict Specific Impulse"):

    prediction = model.predict(input_df)

    st.success(f"Predicted Specific Impulse: {prediction[0]:.2f}")