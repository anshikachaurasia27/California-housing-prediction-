import streamlit as st
import joblib
import pandas as pd
from utils.preprocess import preprocess_input

model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.set_page_config(page_title="Housing Price Predictor", layout="wide")

st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>California Housing Price Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Predict house prices using Machine Learning (Random Forest)</p>",
    unsafe_allow_html=True
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader(" Property Features")
    MedInc = st.slider("Median Income", 0.0, 15.0, 5.0)
    HouseAge = st.slider("House Age", 1, 50, 10)
    AveRooms = st.slider("Average Rooms", 1.0, 10.0, 5.0)
    AveBedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0)

with col2:
    st.subheader("Location Details")
    Population = st.slider("Population", 100, 5000, 1000)
    AveOccup = st.slider("Average Occupancy", 1.0, 10.0, 3.0)
    Latitude = st.slider("Latitude", 32.0, 42.0, 36.0)
    Longitude = st.slider("Longitude", -125.0, -114.0, -120.0)

st.divider()

if st.button(" Predict Price"):
    input_data = {
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude
    }

    processed = preprocess_input(input_data, scaler)
    prediction = model.predict(processed)[0]

    st.success(f" Estimated House Price: ${round(prediction * 100000, 2)}")

    chart_data = pd.DataFrame({
        "Feature": ["Income", "Rooms", "Population"],
        "Value": [MedInc, AveRooms, Population]
    })

    st.subheader(" Input Feature Overview")
    st.bar_chart(chart_data.set_index("Feature"))

   