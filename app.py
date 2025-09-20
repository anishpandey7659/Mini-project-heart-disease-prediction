import streamlit as st
import pickle
import numpy as np
import base64
import requests

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


# ---------- Set background image ----------
def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        .stButton>button {{
            color: white;
            background-color: #4CAF50;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background(r"C:\Users\Lenovo LoQ Laptop\Downloads\Prevention-of-Heart-Disease-in-Women-1024x576.avif")


# ---------- Load trained model ----------
try:
    with open(r"C:\Users\Lenovo LoQ Laptop\Desktop\heart disease predication\logistic_model.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")

# ---------- Load trained scaler ----------
try:
    with open(r"C:\Users\Lenovo LoQ Laptop\Desktop\heart disease predication\Scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    st.success("Scaler loaded successfully!")
except Exception as e:
    st.error(f"Failed to load scaler: {e}")






# ---------- App Title ----------
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict heart disease risk.")


# ---------- Input fields ----------
age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type typical angina:0,atypical angina:1,non-anginal:2,asymptomatic:3", [0, 1, 2,3])
trestbps = st.number_input("Resting Blood Pressure", 50, 250, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
restecg = st.selectbox("Resting ECG normal:0,lv hypertrophy:1,st-t abnormality:2", [0,1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("ST depression", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope of ST segment flat:0,upsloping:1,downsloping:2", [0,1, 2])
ca = st.selectbox("Number of major vessels (0-3)", [0,1, 2, 3])
thal = st.selectbox("Thalassemia (normal:0,reversable defect:1,fixed defect:2)", [0,1, 2])

# ---------- Convert categorical to numeric ----------
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# Predict button
if st.button("Predict"):
    # Convert input to NumPy array
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    try:
        # Scale user input
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)

        # Display result
        if prediction[0] == 0:
            st.success("✅ No Heart Disease detected.")
        else:
            st.error("⚠️ Heart Disease detected! Consult a doctor.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")


