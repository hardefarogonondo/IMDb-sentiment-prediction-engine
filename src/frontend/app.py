import streamlit as st
import requests

st.title('IMDb Sentiment Analysis Engine')

API_URL = "http://127.0.0.1:8000/predict"  # Update with the actual URL if hosted

user_review = st.text_area("Write your movie review here:")

if st.button('Predict'):
    if user_review:
        # Send the review to the API for prediction
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
