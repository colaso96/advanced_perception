# CS 7180 Final Project: Stock Movement Modeling with Minimal Transformer

## (Bronte) Sihan Li, Cole Crescas
### Submission: 2023-12-13
### Use time travel days: None

## Environment

Training tasks are run in a Linux environment with a Nvidia P6000 GPU.

## Requirements

To install dependencies, run:

    pip install -r requirements.txt


## Data

The data used in this project is available from the Optiver Kaggle competition [here](https://www.kaggle.com/competitions/optiver-trading-at-the-close/).  In addition,
our windowed dataset transformation is in a zip file in the data folder

## dataset.py

Contains custom dataset classes for loading data. We define both the original data set and modified time series data set after feature engineering.

## evaluate.py

Contains functions for evaluating model performance and inference time.

## train.py

Main script for training different model architectures. Usage:

    python train.py --model_type <model_type> --batch_size <batch_size> --epochs <epochs> --lr <lr> --val_percent <val_percent> --amp <amp> --save_checkpoint <save_checkpoint>

## models.py

Contains model architectures we custom define for the project, including LSTM, ResCNN and Transformer with different variations.

## xgb_with_matrixload.py

Contains the XGBoost model with the matrix load feature.
