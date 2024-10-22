import streamlit as st
import requests

# Set up Streamlit app title
st.title("Tweet Sentiment Analysis")

# Create an input text box for the tweet
tweet = st.text_area("Enter the tweet you want to analyze:", "", height=150)

# API endpoint URL (adjust this URL based on where your FastAPI is hosted)
api_url = "http://127.0.0.1:8000/predict-sentiment/"

# Create a button to send the request
if st.button("Predict Sentiment"):
    if tweet.strip() == "":
        st.error("Please enter a valid tweet.")
    else:
        # Prepare the payload for the FastAPI request
        payload = {"tweet": tweet}

        # Make a POST request to the FastAPI sentiment analysis API
        try:
            response = requests.post(api_url, json=payload)

            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()
                # Display the sentiment result
                st.success(f"Sentiment: {result['sentiment']}")
                st.write(
                    f"Processing Time: {result['processing_time_seconds']} seconds"
                )
            else:
                st.error(
                    f"Failed to get sentiment. API responded with {response.status_code}."
                )
        except requests.exceptions.RequestException as e:
            st.error(
                f"An error occurred while connecting to the API: {str(e)}")
