from zenml.client import Client
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


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
    print(f"Pipeline artifact [: {latest_run.id}] loaded successfully")
    return {"train": train_loader, "validation": val_loader}
