import logging
import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
from mlflow.tracking import MlflowClient

from utils import get_project_root

# Adding config and src paths
config_path = os.path.join(get_project_root(), 'config')
src_path = os.path.join(get_project_root(), 'src')
sys.path.append(config_path)
sys.path.append(src_path)

import config
import data
import inference

# Initialize logging
logging.basicConfig(level=logging.INFO)


# Define Pydantic model for input validation
class TweetInput(BaseModel):
    tweet: str = Field(
        ...,
        min_length=1,
        max_length=280,
        description="The content of the tweet (1-280 characters)")


# Global variables
client = MlflowClient()
model = None
tokenizer = None


# Lifespan function for managing startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    # Startup logic: Load model and tokenizer
    project_root = os.path.abspath(os.path.join(os.getcwd(), "../"))
    print(project_root)  # Assuming notebook is in the notebooks folder
    mlflow.set_tracking_uri(f"file://{project_root}/mlruns")
    model = inference.load_model_from_registry(model_name=config.model_name)
    logging.info(f"Model {config.model_name} loaded successfully at startup.")
    tokenizer = data.get_tokenizer(config.tokenizer_name)
    logging.info(
        f"Tokenizer {config.tokenizer_name} loaded successfully at startup.")

    # Yield to allow the app to run
    yield

    # Optional: Add any shutdown/cleanup logic if necessary
    logging.info("App shutdown complete.")


# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)


# API route to predict sentiment of a single tweet
@app.post("/predict-sentiment/")
async def predict(tweet_input: TweetInput):
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
    logging.error(f"Validation error processing request: {exc.detail}")
    return {"error": exc.detail}
