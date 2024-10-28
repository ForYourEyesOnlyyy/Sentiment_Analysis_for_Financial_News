# Sentiment Analysis for Financial News

This project leverages a machine learning-based solution to classify the sentiment of financial news tweets as positive, negative, or neutral, empowering investors and analysts to make data-driven decisions. Built using PyTorch and BERT for NLP, the project also integrates ZenML for streamlined data preprocessing and MLflow for robust experiment tracking and model versioning.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Data Processing and Pipelines](#data-processing-and-pipelines)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Web App & Deployment](#web-app--deployment)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Financial news sentiment analysis offers critical insights to investors and analysts, and this project facilitates that analysis using NLP and machine learning. Key functionalities include tokenization, model training, deployment, experiment tracking with MLflow, and data processing using ZenML pipelines. By using BERT as an NLP model layer and tokenizer, the project captures nuanced sentiment in financial tweets.

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
├── setup.sh                  # Setup script for installing dependencies
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

- **Sentiment Classification**: Classifies financial news tweets as positive, negative, or neutral.
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

## Data Processing and Pipelines

Data preprocessing is managed by ZenML, ensuring consistency and ease in data handling across experiments and deployments. The main pipeline, `training_data_pipeline`, performs several key steps to transform raw data into processed data loaders ready for model training. The pipeline is configurable via `TrainingPipelineParams`, allowing for adjustments in batch size, tokenizer, and train-test split ratio.

### Pipeline Steps

1. **Load Data**:
   - The `load` step loads the raw data using the `data.load_data()` function, returning a DataFrame with financial news tweets. This serves as the initial input for subsequent steps.

2. **Preprocess Data**:
   - In `preprocess`, raw data undergoes various cleaning and preprocessing transformations to prepare the text for tokenization. This may include handling missing values, removing irrelevant symbols, and standardizing text formats.

3. **Split Data**:
   - The `split` step divides the data into training and testing sets. The split ratio is defined in `TrainingPipelineParams`, ensuring the pipeline can be easily adjusted for different model validation strategies.

4. **Prepare Dataloaders**:
   - This step, `prepare_dataloaders`, converts the training and testing datasets into data loaders compatible with PyTorch. Using the specified tokenizer from `TrainingPipelineParams`, the data is tokenized and batched. This step returns `train_loader` and `val_loader` dictionaries, ready for use in model training and evaluation.

### Example Pipeline Setup and Execution

```python
from config import config
from training_data_pipeline import training_data_pipeline, TrainingPipelineParams
from src.data import load, preprocess, split, prepare_dataloaders

# Define pipeline parameters
pipeline_params = TrainingPipelineParams(
    batch_size=config.batch_size,
    tokenizer_name=config.tokenizer_name,
    split_ratio=config.split_ratio)

# Initialize pipeline steps
load_instance = load()
preprocess_instance = preprocess()
split_instance = split(params=pipeline_params)
prepare_dataloaders_instance = prepare_dataloaders(params=pipeline_params)

# Create and run pipeline instance
training_data_pipeline_instance = training_data_pipeline(
    load=load_instance,
    preprocess=preprocess_instance,
    split=split_instance,
    prepare_dataloaders=prepare_dataloaders_instance)

training_data_pipeline_instance.run()
```

This pipeline simplifies the end-to-end data preparation process, from loading raw data to generating preprocessed data loaders for model training.

### Extracting Dataloaders from the Latest Pipeline Run

The `extract_latest_loaders` function allows you to access the data loaders from the most recent `training_data_pipeline` run. Using ZenML’s client API, it retrieves the output artifacts from the `prepare_dataloaders` step in the latest pipeline execution, providing the latest `train_loader` and `val_loader` directly for immediate use.

## Model Training, Evaluation, and Tracking

Model training is conducted within the `model_experiments.ipynb` notebook, where various architectures can be implemented for sentiment classification. While

 a BERT-based model is currently set up as a primary example, the flexible framework allows for experimentation with different model architectures. MLflow is used extensively for experiment tracking, versioning, and model management.

### Training Workflow

1. **Data Loader Extraction**:
   - The notebook starts by retrieving data loaders (`train_loader` and `val_loader`) from the most recent ZenML pipeline run, ensuring that training consistently uses the latest preprocessed data.

2. **Model Architecture**:
   - The notebook includes a primary example architecture, `SentimentAnalysisModel`, which uses a pre-trained BERT model with a fully connected layer for classification. However, the notebook’s flexible structure allows for easy integration of additional architectures. 
   - This setup enables experimentation with different models, such as alternative transformer architectures (e.g., RoBERTa, DistilBERT) or custom layers tailored to specific dataset characteristics.
   - The model architecture can be modified and tested in the same notebook, with each variant tracked separately in MLflow, allowing users to compare performance metrics across different model versions.

3. **Training and Validation Functions**:
   - **`train_one_epoch`**: This function performs a forward pass for each batch, computes the loss, and updates model parameters. Training loss per epoch is logged to MLflow for comprehensive tracking.
   - **`val_one_epoch`**: This function evaluates the model on validation data, computing loss and accuracy. Improved validation accuracy prompts the model to be saved as the current best and logs these metrics to MLflow.

4. **MLflow Integration**:
   - MLflow is central to tracking experiments, model versions, and ensuring reproducibility. Key features include:
     - **Parameter Logging**: Logs key hyperparameters such as learning rate, batch size, and number of epochs.
     - **Metric Logging**: Tracks both training and validation losses, as well as validation accuracy, across epochs.
     - **Model Versioning**: After training, models are registered in MLflow’s Model Registry, which manages different versions and provides easy access to previous models.
     - **Champion Model Selection**: Each experiment can automatically assign the “Champion” alias to the model version with the best validation accuracy. This feature ensures that the best model version is easily accessible for further deployment or inference tasks.

### Model Evaluation and Metrics

Evaluation is based on several key metrics to ensure reliable performance:

- **Training Loss**: Monitors model fit on the training dataset.
- **Validation Loss**: Evaluates model performance on unseen data.
- **Validation Accuracy**: Measures the proportion of correct predictions on the validation set.
- **Precision, Recall, and F1-Score** (optional): Additional metrics that can be logged, especially useful when analyzing performance across specific classes.

These metrics, logged with MLflow, provide insights for fine-tuning model performance and comparing the effectiveness of different model architectures.

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
model_name = 'sentiment_analysis_experiment'

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
    register_model(run_id, model_name, "Experimenting with BERT-based architecture")
    update_champion_alias(model_name)
```

### MLflow Model Registry

The MLflow Model Registry manages model version control and provides a seamless way to select the best-performing model as the "Champion." This approach simplifies model comparison and access, ensuring that only the highest-performing version is used for production or deployment tasks. The registry’s alias feature allows easy access to the Champion model, reducing overhead when working with multiple model architectures and versions.

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