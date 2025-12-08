# 📊 Short-Term Forecast with MLOps
This repository provides a structured and reproducible framework for short-term forecasting projects leveraging MLOps principles.

It focuses on predicting short-term behavioral changes and electricity consumption patterns using advanced time-series and graph-based models.

# 🚀 Overview
This project demonstrates how to:

* Structure and organize a data science project within an MLOps pipeline.

* Implement forecasting models integrating neural, graph-based, and traditional time-series methods.

* Ensure reproducibility, scalability, and experiment tracking using MLflow.

## 🔄 MLOps: MLFlow Tracker

The mlflow tracker module provides a wrapper class to simplify the use of MLFlow's tracking functionalities. It allows users to:

* Initializes MLFlow runs.
* Logs parameters, metrics, and artifacts.
* Facilitates experiment management and reproducibility.

## 🧩 Data

The `data/` folder contains scripts related to process time-series data. 
`temporalgnn_data.py` includes a `Dataset` class that can process load and generation timeseries for the purpose of training a graph-based forecasting model using `torch_geometric_temporal`.


## 🧠 Models
The ``models/`` directory contains methodology used for forecasting the short term electricity consumption in an hourly resolution.

#### NeuralProphet
`NeuralProphet` is built on PyTorch and combines Neural Networks and traditional time-series algorithms, inspired by Facebook Prophet and AR-Net.

#### Temporal Graph Neural Network (A3T-GCN)
`temporalgnn_model.py` includes the pytorch_lightning implementation of the A3TGN2 model which combines Graph Convolutional Networks (GCN) and temporal attention mechanisms to capture both spatial dependencie and temporal dependencies for training, validation, and prediction.
