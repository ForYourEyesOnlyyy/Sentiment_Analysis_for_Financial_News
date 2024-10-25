#!/bin/bash

# Default values
PORT="8000"  # Default to 8000 if no port is specified


# Run uvicorn with the specified or default port
echo "Starting FastAPI server on port $PORT..."
uvicorn api:app --reload --port "$PORT"
