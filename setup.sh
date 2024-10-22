#!/bin/bash

# Step 1: Install dependencies
# pip install -r requirements.txt

# Step 2: Set up ZenML stack
bash scripts/create_zenml_stack.sh

# Step 3: Run ZenML server
bash scripts/run_zenml_server.sh

echo "Setup complete!"
echo "You can now run 'bash run.sh' to run the web app."
echo "To stop zenml server, run 'bash scripts/shutdown.sh'."
