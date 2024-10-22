import logging
import os
import sys
import time
import shutil
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
model = None
tokenizer = None


# Copy the mlruns directory from the project root to the app/mlruns directory
def copy_mlruns():
    src_dir = os.path.join(get_project_root(), 'mlruns')
    dest_dir = os.path.join(get_project_root(), 'app', 'mlruns')

    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Copy contents from source to destination
    try:
        # shutil.copytree can't be used here because it requires dest to not exist,
        # so we copy contents manually using copy2
        for item in os.listdir(src_dir):
            src_item = os.path.join(src_dir, item)
            dest_item = os.path.join(dest_dir, item)

            if os.path.isdir(src_item):
                # Recursively copy directories
                shutil.copytree(src_item, dest_item, dirs_exist_ok=True)
            else:
                # Copy files
                shutil.copy2(src_item, dest_item)

        logging.info(
            f"Successfully copied mlruns from {src_dir} to {dest_dir}.")
    except Exception as e:
        logging.error(f"Failed to copy mlruns: {e}")


# Lifespan function for managing startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer

    # Copy mlruns folder at startup
    copy_mlruns()

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(os.path.join(get_project_root(), 'app', 'mlruns'))

    # Load model and tokenizer
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
