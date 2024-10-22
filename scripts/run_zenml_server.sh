#!/bin/bash

# Set environment variable for macOS fork safety
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY='YES'

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
    echo "Starting ZenML server on the default port..."
    zenml up
else
    # Custom port provided, use it
    echo "Starting ZenML server on custom port $PORT..."
    zenml up --port "$PORT"
fi
