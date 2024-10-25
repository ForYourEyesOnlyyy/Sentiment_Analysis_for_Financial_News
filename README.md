# Sentiment Analysis for Financial News

This project leverages a machine learning-based solution to classify the sentiment of financial news articles as positive, negative, or neutral, empowering investors and analysts to make data-driven decisions. Built using PyTorch and BERT for NLP, the project also integrates ZenML for streamlined data preprocessing and MLflow for robust experiment tracking and model versioning.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
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

- **Sentiment Classification**: Classifies financial news articles as positive, negative, or neutral.
- **BERT Integration**: BERT-based tokenization and embedding, integrated as a PyTorch layer for NLP tasks.
- **MLflow Experiment Tracking**: Automatically logs experiments, saving model architecture and weights to the `models` folder, and enables loading the best-performing "champion" model directly from MLflow.
- **ZenML Data Pipelines**: Automates data preprocessing with ZenML pipelines for streamlined training and experimentation.
- **Multiple Deployment Options**: Deployable as a REST API using FastAPI or as a web app using Streamlit.
- **Docker Integration**: Dockerfiles for FastAPI and Streamlit, with Docker Compose for managing both.
- **Automated Setup and Shutdown**: Includes scripts to set up the environment and shut down services.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
---

