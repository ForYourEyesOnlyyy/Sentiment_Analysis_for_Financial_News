import mlflow

model = mlflow.pytorch.load_model(model_uri=f"models:/simple_sentiment_analysis_model/latest")
print(model)