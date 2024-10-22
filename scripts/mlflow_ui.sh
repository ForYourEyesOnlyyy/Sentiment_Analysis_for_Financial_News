#!/bin/bash

# Default values
PORT=""

# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

# Check if a custom port was provided
if [ -z "$PORT" ]; then
    # No port provided, run on the default port
    echo "Starting MLflow UI on the default port (5000)..."
    mlflow ui
else
    # Custom port provided, use it
    echo "Starting MLflow UI on custom port $PORT..."
    mlflow ui --port "$PORT"
fi
