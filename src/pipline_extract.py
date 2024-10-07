from zenml.client import Client

from zenml.steps import BaseStep


def extract_latest_loaders() -> dict:
    # Extract the latest loaders
    client = Client()
    pipline_name = "data_pipeline"
    pipeline = client.get_pipeline(pipline_name)
    latest_run = pipeline.last_run
    loaders_step = latest_run.steps["prepare_dataloaders"]
    artifact = loaders_step.output.load()
    train_loader = artifact["train"]
    val_loader = artifact["validation"]
    return {"train": train_loader, "validation": val_loader}
