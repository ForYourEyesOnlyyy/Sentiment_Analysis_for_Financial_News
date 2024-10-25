#!/bin/bash

# Set environment variable for macOS fork safety (only needed for macOS)
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY='YES'

# Default Streamlit port
STREAMLIT_PORT="8501"

# Define the paths to your FastAPI and Streamlit app files
FASTAPI_APP="deployment.api:app"  # FastAPI app is in the 'app' folder, so we use 'app.api:app'
STREAMLIT_APP="deployment/app.py"  # Streamlit app is in the 'app' folder

# # Navigate to the project root (assuming this script is executed from the 'scripts' folder)
# cd "$(dirname "$0")/.." || exit 1

# Parse command-line options for custom Streamlit port
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --port)
            STREAMLIT_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

# Function to run FastAPI
run_fastapi() {
    echo "Starting FastAPI server on port 8000..."
    uvicorn "$FASTAPI_APP" --reload --port 8000 &  # Runs FastAPI in the background
    FASTAPI_PID=$!  # Store the FastAPI process ID to kill later
}

# Function to run Streamlit
run_streamlit() {
    echo "Starting Streamlit app on port $STREAMLIT_PORT..."
    streamlit run "$STREAMLIT_APP" --server.port "$STREAMLIT_PORT" &  # Runs Streamlit in the background
    STREAMLIT_PID=$!  # Store the Streamlit process ID to kill later
}

# Function to stop the background processes
cleanup() {
    echo "Stopping FastAPI and Streamlit servers..."
    kill $FASTAPI_PID
    kill $STREAMLIT_PID
}

# Trap to ensure cleanup on script exit (Ctrl+C or termination)
trap cleanup EXIT

# Run FastAPI and Streamlit
run_fastapi
run_streamlit

# Wait for the background processes to finish
wait
