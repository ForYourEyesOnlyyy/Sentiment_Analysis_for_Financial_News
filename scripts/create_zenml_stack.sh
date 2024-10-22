#!/bin/bash

# Function to check if a command exists
command_exists () {
    type "$1" &> /dev/null ;
}

# Check if ZenML is installed
if ! command_exists zenml; then
    echo "ZenML is not installed. Installing ZenML..."
    pip install zenml
else
    echo "ZenML is already installed."
fi

# Get the current working directory (absolute path)
PROJECT_ROOT=$(pwd)

# Check if the stack already exists
STACK_NAME="sentiment_analysis_stack"
if zenml stack list | grep -q "$STACK_NAME"; then
    echo "ZenML stack '$STACK_NAME' already exists. No changes made."
else
    echo "Creating ZenML stack '$STACK_NAME'..."

    # Register artifact store with the absolute path
    zenml artifact-store register sentiment_analysis_artifact_store --flavor=local --path="${PROJECT_ROOT}/services/zenml_artifacts"

    # Register orchestrator
    zenml orchestrator register sentiment_analysis_orchestrator --flavor=local

    # Register and set the stack
    zenml stack register "$STACK_NAME" \
        -a sentiment_analysis_artifact_store \
        -o sentiment_analysis_orchestrator

    # Set the stack as active
    zenml stack set "$STACK_NAME"
fi

# Describe the active stack
zenml stack describe
