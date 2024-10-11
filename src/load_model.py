import mlflow

def load_model_from_registry(model_name, version='latest'):
    model = mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{version}")
    return model


