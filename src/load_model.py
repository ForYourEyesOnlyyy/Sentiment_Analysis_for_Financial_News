import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

def load_model_from_registry(model_name, version="latest"):
    if version == "champion":
        global client
        champion_version = client.get_model_version_by_alias(name=model_name, alias="champion").version
        model = mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{champion_version}")
    else:
        model = mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{version}")

    print(f"Model {model_name} loaded successfully")
    return model