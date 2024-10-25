# Sentiment Analysis for Financial News

This project delivers a machine learning-based solution for sentiment analysis of financial news articles, classifying each as positive, negative, or neutral. Aimed at supporting investors and analysts, the project provides both a REST API and web app deployment via FastAPI, Streamlit, Docker, and ZenML for efficient sentiment analysis in financial contexts.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

Financial news sentiment analysis offers critical insights to investors and analysts, and this project facilitates that analysis using NLP and machine learning. The goal is to streamline sentiment classification, enhancing financial decision-making by providing accessible sentiment insights for market events.

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

---

