import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('delivery_model.pkl')

encoders = joblib.load('encoders.pkl')

# Title
st.title("Food Delivery Time Prediction")

st.write("Predict estimated delivery time using ML")

# User Inputs

distance = st.number_input(
    "Distance (km)",
    min_value=0.0
)

weather = st.selectbox(
    "Weather",
    ['Sunny', 'Rainy', 'Foggy', 'Windy']
)

traffic = st.selectbox(
    "Traffic Level",
    ['Low', 'Medium', 'High']
)

time_of_day = st.selectbox(
    "Time of Day",
    ['Morning', 'Afternoon', 'Evening', 'Night']
)

vehicle = st.selectbox(
    "Vehicle Type",
    ['Bike', 'Scooter', 'Car']
)

prep_time = st.number_input(
    "Preparation Time (mins)",
    min_value=0.0
)

experience = st.number_input(
    "Courier Experience (yrs)",
    min_value=0.0
)

# Predict Button

if st.button("Predict Delivery Time"):

    input_data = pd.DataFrame({

        'distance_km': [distance],

        'weather': [weather],

        'traffic_level': [traffic],

        'time_of_day': [time_of_day],

        'vehicle_type': [vehicle],

        'preparation_time_min': [prep_time],

        'courier_experience_yrs': [experience]
    })

    categorical_columns = [

        'weather',
        'traffic_level',
        'time_of_day',
        'vehicle_type'
    ]

    # Encoding

    for col in categorical_columns:

        input_data[col] = encoders[col].transform(
            input_data[col]
        )

    # Prediction

    prediction = model.predict(input_data)

    st.success(
        f"Estimated Delivery Time: {prediction[0]:.2f} mins"
    )