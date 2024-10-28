Here’s the updated README with the additional note in the setup instructions about using `shutdown.sh` to stop background processes:

---

# Sentiment Analysis for Financial News

This project leverages a machine learning-based solution to classify the sentiment of financial news articles as positive, negative, or neutral, empowering investors and analysts to make data-driven decisions. Built using PyTorch and BERT for NLP, the project also integrates ZenML for streamlined data preprocessing and MLflow for robust experiment tracking and model versioning.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Data Processing and Pipelines](#data-processing-and-pipelines)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Deployment](#deployment)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Financial news sentiment analysis offers critical insights to investors and analysts, and this project facilitates that analysis using NLP and machine learning. Key functionalities include tokenization, model training, deployment, experiment tracking with MLflow, and data processing using ZenML pipelines. By using BERT as an NLP model layer and tokenizer, the project captures nuanced sentiment in financial text.

## Project Structure

```plaintext
Sentiment-Analysis-for-Financial-News/
├── LICENSE
├── README.md
├── config                    # Configuration files (model name, device type, etc.)
├── data                      # Data directory
│   ├── dvc                   # DVC data management files
│   └── processed             # Processed data files
├── deployment                # Deployment resources
│   ├── Dockerfile.fastapi    # Dockerfile for FastAPI service
│   ├── Dockerfile.streamlit  # Dockerfile for Streamlit app
│   ├── api.py                # FastAPI code for API
│   ├── app.py                # Streamlit app code for front-end
│   ├── docker-compose.yml    # Docker Compose file to coordinate services
│   └── requirements          # Requirements for deployment
│       ├── fastapi-requirements.txt
│       └── streamlit-requirements.txt
├── models                    # Model architectures and saved weights
├── notebooks                 # Jupyter notebooks for exploration, preprocessing, and model training
│   ├── data_exploration.ipynb          # Initial exploration and analysis of data
│   ├── feature_engineering.ipynb       # Feature engineering for model input
│   └── model_experiments.ipynb         # Model experimentation and evaluation
├── pipelines                 # ZenML pipeline scripts for preprocessing and training
│   ├── extract_training_data.py        # Script to extract training data
│   └── training_data_pipeline.py       # Pipeline for processing training data
├── requirements.txt          # General dependencies
├── run.sh                    # Script to run the web app
├── setup.sh                  # Setup script for installing dependencies and ZenML stack
├── shutdown.sh               # Script to stop background services
├── scripts                   # Bash scripts for running and setting up the project
│   ├── create_zenml_stack.sh          # Script to create ZenML stack
│   ├── mlflow_ui.sh                   # Script to launch MLflow UI
│   ├── run_app.sh                     # Script to run the web application
│   ├── run_fastapi.sh                 # Script to run FastAPI
│   └── run_zenml_server.sh            # Script to start ZenML server
├── services                  # Additional service configuration and code
└── src                       # Core source code for the app
    ├── data.py               # Data processing and preparation code
    └── inference.py          # Code for making predictions with trained models
```

## Features

- **Sentiment Classification**: Classifies financial news articles as positive, negative, or neutral.
- **BERT Integration**: BERT-based tokenization and embedding, integrated as a PyTorch layer for NLP tasks.
- **MLflow Experiment Tracking**: Automatically logs experiments, saving model architecture and weights to the `models` folder, and enables loading the best-performing "champion" model directly from MLflow.
- **ZenML Data Pipelines**: Automates data preprocessing with ZenML pipelines for streamlined training and experimentation.
- **Multiple Deployment Options**: Deployable as a REST API using FastAPI or as a web app using Streamlit.
- **Docker Integration**: Dockerfiles for FastAPI and Streamlit, with Docker Compose for managing both.
- **Automated Setup and Shutdown**: Includes scripts to set up the environment and shut down services.

## Setup Instructions

### Prerequisites

- **Python** 3.11+
- **Docker** (for deployment)
- **pip** (Python package installer)

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/Sentiment-Analysis-for-Financial-News.git
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

To launch the web app (FastAPI backend and Streamlit UI), follow these steps:
1. Run the setup script as described above.
2. Run the following command to start the app:
    ```bash
    bash run.sh
    ```

## Data Processing and Pipelines

Data preprocessing is managed by ZenML, which includes:
- **ZenML Data Pipeline**: Located in `pipelines`, the ZenML pipeline includes scripts for data extraction and preprocessing, allowing for easy data handling and consistent processing across experiments.
  - `extract_training_data.py`: Script to extract training data from the source.
  - `training_data_pipeline.py`: Main pipeline for data preprocessing and preparing training-ready data.

Here’s an enhanced README section with additional details on model evaluation, metrics, and MLflow’s role in the training and tracking process:

---

## Model Training, Evaluation, and Tracking

Model training is conducted within the `model_experiments.ipynb` notebook, where we use a BERT-based architecture to perform sentiment classification. MLflow plays a central role in experiment tracking, model versioning, and ensuring reproducibility, while evaluation metrics allow us to monitor and improve model performance.

### Training Workflow

1. **Data Loader Extraction**:
   - The notebook starts by retrieving data loaders (`train_loader` and `val_loader`) from the most recent ZenML pipeline run, ensuring that training always leverages the latest preprocessed data.

2. **Model Architecture**:
   - The custom `SentimentAnalysisModel` is based on a pre-trained BERT model, with a fully connected layer added to handle multi-class classification (positive, neutral, negative). The model also incorporates dropout to mitigate overfitting.

3. **Training and Validation Functions**:
   - **`train_one_epoch`**: This function performs a forward pass for each batch, computes the loss, and updates model parameters. It logs training loss per epoch to MLflow.
   - **`val_one_epoch`**: This function evaluates the model on validation data, computing both loss and accuracy. If the validation accuracy improves, the function saves the model as the current best and logs the metrics to MLflow.

4. **MLflow Integration**:
   - MLflow is used extensively in this project for experiment tracking and model management, enabling seamless logging, model versioning, and "Champion" model selection.
     - **Parameter Logging**: Key hyperparameters, such as learning rate and number of epochs, are logged at the start of each experiment.
     - **Metric Logging**: Training and validation losses are logged at each epoch, alongside validation accuracy.
     - **Model Versioning**: At the end of training, the model is registered in MLflow’s Model Registry. This provides version control and simplifies access to previous models.
     - **Champion Model Selection**: Upon registration, the notebook designates the highest-performing model as the “Champion” model. MLflow’s alias feature enables easy retrieval of this champion model for deployment and inference.

### Model Evaluation and Metrics

Model performance is evaluated on multiple metrics to ensure robustness and reliability:

- **Training Loss**: Indicates how well the model fits the training data over each epoch.
- **Validation Loss**: Measures model performance on unseen data, crucial for identifying overfitting or underfitting.
- **Validation Accuracy**: A primary indicator of the model’s classification performance, this metric shows the proportion of correct predictions in the validation set.
- **Precision, Recall, and F1-Score** (optional): These metrics can be logged for further performance analysis, especially if a more detailed evaluation on specific classes is required.

By monitoring these metrics, we can track model improvements across experiments and adjust hyperparameters or model architecture as needed.

### Example Workflow for Model Training and Evaluation

```python
from src.data import SentimentAnalysisModel, train_one_epoch, val_one_epoch, register_model, update_champion_alias
import mlflow
import torch.optim as optim
import os

# Model and training configuration
epochs = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 2e-5
model_name = 'simple_sentiment_analysis_model'

# Initialize model, criterion, and optimizer
model = SentimentAnalysisModel(bert_model_name='bert-base-uncased', num_labels=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Tracking with MLflow
with mlflow.start_run():
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("epochs", epochs)
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        best_so_far = val_one_epoch(model, val_loader, criterion, device, epoch, best_so_far, model_name)
    # Register and assign Champion model
    run_id = mlflow.active_run().info.run_id
    register_model(run_id, model_name, "BERT-based model with dropout and one fully connected layer")
    update_champion_alias(model_name)
```

### MLflow Model Registry

The MLflow Model Registry simplifies model version control and selection by designating a champion model based on validation performance. This process involves:
- **Model Registration**: Each completed training run registers a new model version.
- **Champion Model Alias**: The model version with the highest validation accuracy is assigned the “Champion” alias, providing a standardized and efficient way to retrieve the best-performing model for deployment or inference.

This approach ensures that only the best model is used for production, and it enables quick experimentation without losing track of previous high-performing models. MLflow’s tracking and registry tools allow for robust experimentation, reproducibility, and scalable model management.

Here’s an updated README section for the web app and deployment, incorporating your details about FastAPI, Streamlit, Docker configuration, and Docker Compose networking:

---

## Web App & Deployment

The project includes a web application with both a REST API (powered by FastAPI) and a user-friendly UI (built with Streamlit) for sentiment analysis of financial news or tweets. The entire application is containerized and managed with Docker, enabling streamlined deployment and efficient resource management.

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

response = requests.post("http://localhost:8000/predict", json={"text": "Sample financial news article"})
print(response.json())  # {'sentiment': 'positive'}
```

### Streamlit

Access the Streamlit app at `http://localhost:8501` to interactively classify news articles.

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

This README now includes the additional shutdown instructions for a complete and clean setup. Let me know if there's more you'd like to add!


Here's the modified table of contents to reflect the details about the web app, deployment, and other sections based on the content we discussed:

---

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Data Processing and Pipelines](#data-processing-and-pipelines)
- [Model Training, Evaluation, and Tracking](#model-training-evaluation-and-tracking)
- [Web App & Deployment](#web-app--deployment)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license) 

---

This table of contents is now organized to capture the full workflow, from data processing through deployment and usage.


{'tweet': 'Sample financial news tweet', 'sentiment': 'positive', 'processing_time_seconds': 0.06371}