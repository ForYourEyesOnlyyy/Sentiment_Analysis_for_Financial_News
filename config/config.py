"""
config.py

This configuration file defines constants and parameters used throughout the project for 
data loading, preprocessing, model setup, and training configuration. The values here 
are referenced across modules to ensure consistency and easy adjustments.

Attributes:
    # Data Paths and Columns
    data_path (str): The file path to the processed dataset.
    label_column (str): The column name representing sentiment labels in the dataset.
    text_column (str): The column name containing tweet text.
    has_source_column (str): The column name indicating the presence of source links.
    sentiments (dict): A dictionary mapping label IDs to sentiment names.

    # Dataset and Preprocessing Parameters
    batch_size (int): Number of samples per batch for training and validation.
    split_ratio (float): Ratio for splitting the data into training and validation sets.
    max_length (int): Maximum sequence length for tokenized text.

    # Model Parameters
    tokenizer_name (str): The name of the tokenizer used for preprocessing.
    model_name (str): Name of the model used for sentiment analysis.
    model_class (obj): Instance of the model class used for sentiment analysis.
    model_weights_path (str): File path for saving and loading the model's weights.
    
    # Device Configuration
    device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
"""

# DATA
data_path = "data/processed/twitter-financial-news-sentiment/samples/sample1.csv"
label_column = "label"
text_column = "text"
has_source_column = "has_source"
sentiments = {0: "Negative", 1: "Positive", 2: "Neutral"}
labels = {"Negative": 0, "Positive": 1, "Neutral": 2}

# DATASET
batch_size = 32
split_ratio = 0.33
max_length = 100

# MODELS
tokenizer_name = 'bert-base-uncased'
model_name = 'simple_sentiment_analysis_model'
from models.simple_sentiment_analysis_model.simple_sentiment_analysis_model import SentimentAnalysisModel

model_class = SentimentAnalysisModel()
model_weights_path = f'models/{model_name}/model_weights.pth'

# DEVICE
device = 'cpu'
