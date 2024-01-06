# Import Libraries
import requests
import streamlit as st

# Initialization
API_URL = 'http://backend:8000/predict'

# Main UI
st.title("IMDb Sentiment Analysis Engine")
user_review = st.text_area("Write your movie review here:")

# Predict Sentiment Button
if st.button("Predict"):
    if user_review:
        response = requests.post(API_URL, json={"review": user_review})
        if response.status_code == 200:
            data = response.json()
            result = data["result"]
            confidence = data["confidence"]
            st.write(f"Predicted Sentiment: {result} (Confidence: {confidence:.2f})")
        else:
            st.write("Failed to get a response from the model.")
    else:
        st.write("Please enter a review for prediction.")
