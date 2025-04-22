import streamlit as st
import pandas as pd
from predictor import Predictor

# Inisialisasi predictor
predictor = Predictor('RF_class.pkl', 'training_columns.pkl')
columns_to_encode = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']

st.title("Hotel Booking Cancellation Prediction")

st.markdown("### Masukkan informasi pemesanan:")

# Form input
no_of_adults = st.number_input("Number of Adults", 0, 10, 2)
no_of_children = st.number_input("Number of Children", 0, 10, 0)
lead_time = st.number_input("Lead Time", 0, 1000, 100)
arrival_year = st.selectbox("Arrival Year", [2021, 2022, 2023])
arrival_month = st.selectbox("Arrival Month", list(range(1, 13)))
arrival_date = st.selectbox("Arrival Date", list(range(1, 32)))
no_of_weekend_nights = st.slider("Weekend Nights", 0, 7, 1)
no_of_week_nights = st.slider("Week Nights", 0, 14, 2)
type_of_meal_plan = st.selectbox("Meal Plan", ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
required_car_parking_space = st.selectbox("Parking Space Required", [0, 1])
room_type_reserved = st.selectbox("Room Type", ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])
market_segment_type = st.selectbox("Market Segment", ['Online', 'Offline', 'Corporate', 'Complementary', 'Aviation'])
repeated_guest = st.selectbox("Repeated Guest", [0, 1])
no_of_previous_cancellations = st.number_input("Previous Cancellations", 0, 10, 0)
no_of_previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", 0, 50, 1)
avg_price_per_room = st.number_input("Average Price per Room", 0.0, 10000.0, 120.0)
no_of_special_requests = st.slider("Special Requests", 0, 5, 0)

if st.button("Predict"):
    user_input = pd.DataFrame([{
        'no_of_adults': no_of_adults,
        'no_of_children': no_of_children,
        'lead_time': lead_time,
        'arrival_year': arrival_year,
        'arrival_month': arrival_month,
        'arrival_date': arrival_date,
        'no_of_weekend_nights': no_of_weekend_nights,
        'no_of_week_nights': no_of_week_nights,
        'type_of_meal_plan': type_of_meal_plan,
        'required_car_parking_space': required_car_parking_space,
        'room_type_reserved': room_type_reserved,
        'market_segment_type': market_segment_type,
        'repeated_guest': repeated_guest,
        'no_of_previous_cancellations': no_of_previous_cancellations,
        'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
        'avg_price_per_room': avg_price_per_room,
        'no_of_special_requests': no_of_special_requests
    }])

    prepared = predictor.prepare_input(user_input, columns_to_encode)
    result = predictor.predict(prepared)
    label = "Not Canceled" if result[0] == 1 else "Canceled"

    st.success(f"Hasil Prediksi: **{label}**")
