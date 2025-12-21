import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------ Load Model & Scaler ------------------
try:
    model = joblib.load('fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model or Scaler not found. Ensure 'fraud_model.pkl' and 'scaler.pkl' exist.")
    st.stop()

# ------------------ UI ------------------
st.title("💳 Credit Card Fraud Detection Project")
st.header("Introduction and Project Goals")

st.markdown("""
This project uses a machine learning model to detect fraudulent credit card transactions.
The dataset is **highly imbalanced**, so special preprocessing techniques were applied.
""")

# ------------------ Model Section ------------------
st.header("Machine Learning Model & Prediction")

st.subheader("Predict a Transaction")

with st.form("prediction_form"):
    st.write("Enter transaction details:")

    time = st.slider("Time (seconds since first transaction)", 0, 172792, 50000)
    amount = st.number_input("Transaction Amount", min_value=0.0, max_value=25691.16, value=50.0)

    v4 = st.number_input("V4", value=1.0)
    v11 = st.number_input("V11", value=0.0)

    submitted = st.form_submit_button("Predict")

    if submitted:
        # ------------------ Prepare V-features ------------------
        raw_input = {f'V{i}': 0.0 for i in range(1, 29)}
        raw_input['V4'] = v4
        raw_input['V11'] = v11

        # ------------------ Scale Amount & Time ------------------
        amount_scaled = scaler.transform([[amount]])[0][0]
        time_scaled = scaler.transform([[time]])[0][0]

        # ------------------ Create Final Input DataFrame ------------------
        input_df = pd.DataFrame([raw_input])
        input_df['Amount_scaled'] = amount_scaled
        input_df['Time_scaled'] = time_scaled

        # Ensure correct column order
        expected_columns = [f'V{i}' for i in range(1, 29)] + ['Amount_scaled', 'Time_scaled']
        input_df = input_df[expected_columns]

        # ------------------ Prediction ------------------
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)[:, 1]

        # ------------------ Output ------------------
        if prediction[0] == 1:
            st.error(f"🚨 FRAUDULENT TRANSACTION (Confidence: {prediction_proba[0]:.2f})")
        else:
            st.success(f"✅ NOT FRAUDULENT (Confidence: {1 - prediction_proba[0]:.2f})")

# ------------------ Performance Summary ------------------
st.subheader("Model Performance Summary")
st.markdown("""
- **AUC-ROC:** ~0.95  
- **High Recall:** Detects most fraud cases  
- **High Precision:** Fewer false alarms  

This demonstrates strong fraud detection capability on imbalanced data.
""")

# ------------------ Conclusion ------------------
st.header("Conclusion")
st.markdown("""
This application demonstrates an end-to-end fraud detection system using:
- Feature scaling
- Class imbalance handling
- Machine learning inference
- Streamlit UI

Correct feature alignment is critical for accurate predictions.
""")
