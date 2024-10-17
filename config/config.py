# DATA
data_path = "data/processed/twitter-financial-news-sentiment/samples/sample1.csv"
label_column = "label"
text_column = "text"
has_source_column = "has_source"
sentiments = {0: "Negative", 1: "Neutral", 2: "Positive"}

# DATASET
batch_size = 32
split_ratio = 0.33
max_length = 100

# MODELS
tokenizer_name = 'bert-base-uncased'
model_name = 'simple_sentiment_analysis_model'