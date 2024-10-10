from zenml.client import Client

# Get the ZenML client
client = Client()

# Get the local store path
local_store_path = client.zen_store

print(f"ZenML Local Store Path: {local_store_path}")