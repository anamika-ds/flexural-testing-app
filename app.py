import streamlit as st
import joblib
import pandas as pd

# Load the model and scaler
model = joblib.load('multitarget_random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app title
st.title("Flexural Stress, Strain, and Extension Predictor")


# Explanation of Flexural Testing
st.markdown("""
### What is Flexural Testing?
Flexural testing, also known as bend testing, is a mechanical test used to determine the strength and stiffness of materials when subjected to bending forces. 
It is commonly used to evaluate materials like metals, polymers, ceramics, and composites. The test involves applying a load to a specimen until it bends or fractures, 
and measuring the resulting stress, strain, and extension.

- **Flexural Stress (MPa):** The maximum stress experienced by the material at its outermost fibers during bending.
- **Flexural Strain (%):** The deformation or elongation of the material as a result of the applied stress.
- **Extension (mm):** The amount by which the material stretches or extends under the applied load.

This app predicts the flexural stress, strain, and extension based on the applied load (in kN) using a pre-trained machine learning model.
""")

# Input field for Load(kN)
load_kN = st.number_input("Enter Load (kN):", min_value=0.0, max_value=100.0, value=5.0, step=0.1)

# Predict button
if st.button("Predict"):
    # Scale the input
    input_scaled = scaler.transform([[load_kN]])

    # Make predictions
    predictions = model.predict(input_scaled)

    # Display predictions
    st.subheader("Predictions:")
    st.write(f"Extension (mm): {predictions[0][0]:.2f}")
    st.write(f"Flexural Stress (MPa): {predictions[0][1]:.2f}")
    st.write(f"Flexural Strain (%): {predictions[0][2]:.2f}")
    
