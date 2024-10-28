# Sentiment Analysis for Financial News

This project leverages a machine learning-based solution to classify the sentiment of financial news tweets as positive, negative, or neutral, empowering investors and analysts to make data-driven decisions. Built using PyTorch and BERT for NLP, the project also integrates ZenML for streamlined data preprocessing and MLflow for robust experiment tracking and model versioning.

## Table of Contents

- [Overview](#overview)
- [Setup Instructions](#setup-instructions)
- [Web App & Deployment](#web-app--deployment)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Financial news sentiment analysis offers critical insights to investors and analysts, and this project facilitates that analysis using NLP and machine learning. Key functionalities include tokenization, model training, deployment, experiment tracking with MLflow, and data processing using ZenML pipelines. By using BERT as an NLP model layer and tokenizer, the project captures nuanced sentiment in financial tweets.

## Setup Instructions

### Prerequisites

- **Python** 3.11+
- **Docker** (for deployment)
- **pip** (Python package installer)

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/ForYourEyesOnlyyy/Sentiment_Analysis_for_Financial_News.git
    cd Sentiment-Analysis-for-Financial-News
    ```

2. **Create a Virtual Environment**:
    - Create an empty virtual environment with Python 3.11+:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Set Up the Project**:
    - Run the setup script to install requirements, create the ZenML stack, and set it as current:
    ```bash
    bash setup.sh
    ```

4. **Run MLflow UI** (Optional):
    - To launch the MLflow UI for tracking experiments, run:
    ```bash
    bash scripts/mlflow_ui.sh
    ```

5. **Shutdown**:
    - Once done with the project, run the following command to stop all background processes (such as the ZenML server):
    ```bash
    bash shutdown.sh
    ```

After completing these steps, you're ready to start using or contributing to the project.

### Running the Web App

To launch the web app (FastAPI backend and Streamlit UI), follow one of the two methods below:

#### Option 1: Run with Docker

1. Make sure Docker is installed and running on your system.
2. Use Docker Compose to build and run the application:
    ```bash
    docker-compose -f deployment/docker-compose.yml up --build
    ```

#### Option 2: Run with Bash Scripts

1. Use the following command to start the app:
    ```bash
    bash run.sh
    ```

## Web App & Deployment

The project includes a web application with both a REST API (powered by FastAPI) and a user-friendly UI (built with Streamlit) for sentiment analysis of financial news tweets. The entire application is containerized and managed with Docker, enabling streamlined deployment and efficient resource management.

### REST API with FastAPI

- **FastAPI** is used to create a REST API endpoint that accepts a string input (e.g., a tweet or news excerpt) and returns its sentiment. Upon receiving a request:
  - The API performs **model inference** to classify the sentiment as positive, neutral, or negative.
  - **Inference time** is also calculated and returned with the result to provide insight into the model’s performance in real-time applications.

### User Interface with Streamlit

- **Streamlit** serves as a front-end interface for users to interact with the model. Users can input text and receive sentiment analysis results directly, making the model accessible without needing programming knowledge or API integration.

### Docker & Deployment Setup

The project is containerized with Docker for consistent and lightweight deployment. The Docker setup includes:

1. **Dockerfiles**:
   - **API Dockerfile**: Builds an image for the FastAPI-based backend, allowing it to handle sentiment analysis requests.
   - **App Dockerfile**: Builds an image for the Streamlit UI, enabling user-friendly interaction with the model.

2. **Docker Compose**:
   - A `docker-compose.yml` file is used to build and run both the API and UI containers together.
   - For efficiency and to keep container sizes minimal, a **shared project folder** is mounted in Docker Compose, allowing both containers to access necessary project files without duplicating data.
   - **Networking**: Docker Compose creates a network that links the API and UI containers, allowing Streamlit to communicate with FastAPI for real-time sentiment analysis.

### Running the Web App

To run the web app and API together, use Docker Compose:
```bash
docker-compose -f deployment/docker-compose.yml up --build
```

This command builds and launches both containers, setting up the FastAPI backend and Streamlit UI to work seamlessly together in a single network. The setup allows the API to process requests from the Streamlit UI efficiently and returns sentiment results and inference times directly to the user. 

This containerized deployment setup ensures that the web app is easily scalable and can be deployed consistently across different environments, supporting efficient and responsive sentiment analysis applications.

## Usage

### FastAPI

Once deployed, the API accepts HTTP POST requests for sentiment predictions:
```python
import requests

response = requests.post("http://localhost:8000/predict", json={"tweet": "Sample financial news tweet"})
print(response.json())  # {'tweet': 'Sample financial news tweet', 'sentiment': 'positive', 'processing_time_seconds': 0.06371}
```

### Streamlit

Access the Streamlit app at `http://localhost:8501` to interactively classify tweets.

## Contributing

Contributions are welcome! Follow these steps to contribute:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/new-feature`.
3. Commit your changes: `git commit -m 'Add new feature'`.
4. Push to the branch: `git push origin feature/new-feature`.
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

--- 