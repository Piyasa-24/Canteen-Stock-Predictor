import streamlit as st
import pandas as pd
import pickle
import calendar
from datetime import datetime


# Load the model and encoders
model = pickle.load(open('model.pkl', 'rb'))
le_item = pickle.load(open('le_item.pkl', 'rb'))
le_day = pickle.load(open('le_day.pkl', 'rb'))
le_day.fit(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])


# Streamlit UI
st.title("🍽️ Canteen Sales Predictor")
st.markdown("Predict how many plates/cups to prepare today!")

# Inputs
date_input = st.date_input("📅 Choose Date", datetime.today())
item_input = st.selectbox("🍴 Choose Item", ['Samosa', 'Tea'])

# Process input
day_name = calendar.day_name[date_input.weekday()]
item_encoded = le_item.transform([item_input])[0]
day_encoded = le_day.transform([day_name])[0]

# Predict
if st.button("Predict Quantity"):
    prediction = model.predict([[item_encoded, day_encoded]])[0]
    st.success(f"Estimated Quantity: **{int(prediction)}** units")
