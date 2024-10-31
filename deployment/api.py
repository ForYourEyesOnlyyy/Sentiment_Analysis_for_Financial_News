"""
api.py

This FastAPI application provides an API endpoint for tweet sentiment analysis. 
The API accepts a tweet as input and returns a sentiment prediction along with 
the processing time. The application loads a pre-trained model and tokenizer at startup, 
and it uses a custom error handler to manage input validation errors.

Classes:
    TweetInput: Pydantic model for input validation, ensuring tweet text is within 1-280 characters.

Global Variables:
    model (SentimentAnalysisModel): The sentiment analysis model loaded at startup.
    tokenizer (AutoTokenizer): Tokenizer instance associated with the model.

Endpoints:
    /predict-sentiment/ (POST): Accepts a tweet and returns the predicted sentiment.

Attributes:
    model (SentimentAnalysisModel): The sentiment analysis model used for predictions.
    tokenizer (AutoTokenizer): Tokenizer used for preprocessing tweets before sentiment prediction.

Usage:
    Run this app to serve sentiment predictions via an API. The model and tokenizer are loaded 
    at startup for efficient processing, and request processing time is logged for each prediction.
"""

import logging
import time

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch

from config import config
from src import data
from src import inference

# Initialize logging
logging.basicConfig(level=logging.INFO)


# Define Pydantic model for input validation
class TweetInput(BaseModel):
    """Pydantic model for validating tweet input.

    Attributes:
        tweet (str): The content of the tweet, with a minimum length of 1 
                     and a maximum length of 280 characters.
    """
    tweet: str = Field(
        ...,
        min_length=1,
        max_length=280,
        description="The content of the tweet (1-280 characters)")


# Global variables for model and tokenizer instances
model = None
tokenizer = None


# Lifespan function for managing startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the model and tokenizer at app startup and handles cleanup on shutdown.

    Args:
        app (FastAPI): The FastAPI app instance.

    Yields:
        None: Allows the FastAPI app to run, with resources initialized for the app's lifetime.

    Initializes:
        model (SentimentAnalysisModel): The pre-trained sentiment analysis model.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
    """
    global model, tokenizer

    # Load model and tokenizer at startup
    model = config.model_class
    model.load_state_dict(
        torch.load(config.model_weights_path, map_location=config.device))
    logging.info(f"Model {config.model_name} loaded successfully at startup.")

    tokenizer = data.get_tokenizer(config.tokenizer_name)
    logging.info(
        f"Tokenizer {config.tokenizer_name} loaded successfully at startup.")

    # Yield to allow the app to run
    yield

    # Optional: Add any shutdown/cleanup logic if necessary
    logging.info("App shutdown complete.")


# Initialize FastAPI app with lifespan management
app = FastAPI(lifespan=lifespan)


# API route to predict sentiment of a single tweet
@app.post("/predict-sentiment/")
async def predict(tweet_input: TweetInput):
    """Predicts the sentiment of a tweet.

    Args:
        tweet_input (TweetInput): The tweet content submitted for sentiment analysis.

    Returns:
        dict: A dictionary containing the original tweet, the predicted sentiment, 
              and the processing time in seconds.
    
    Logs:
        Information about the received tweet and processing time for the request.
    """
    # Start time to measure request processing duration
    start_time = time.time()

    # Logging the received tweet
    logging.info(f"Received tweet for sentiment analysis: {tweet_input.tweet}")

    # Call the prediction function (includes internal preprocessing)
    sentiment = inference.predict_sentiment(tweet_input.tweet, tokenizer,
                                            model)

    # Calculate request processing duration
    processing_time = time.time() - start_time

    # Return the result with metadata
    return {
        "tweet": tweet_input.tweet,
        "sentiment": sentiment,
        "processing_time_seconds": round(processing_time, 4)
    }


# Custom error handler for HTTP 422 validation errors
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request, exc):
    """Handles validation errors and logs them.

    Args:
        request: The HTTP request that caused the error.
        exc (HTTPException): The exception instance containing error details.

    Returns:
        dict: A dictionary containing the error message for client feedback.
    
    Logs:
        Error details when input validation fails.
    """
    logging.error(f"Validation error processing request: {exc.detail}")
    return {"error": exc.detail}
