# Movie Profit Prediction with Neural Network

This project implements a simple neural network using PyTorch to predict movie profit ratios based on various categorical and numerical features. It includes data preprocessing, training, evaluation, and permutation-based feature importance analysis.

## Overview

- Loads and preprocesses movie dataset (`Movies_Parties_cat_BIG.csv`), including expanding categorical columns (e.g., production companies).
- One-hot encodes categorical features such as actors, genres, and production companies.
- Trains a fully connected neural network to predict the `profit_ratio`.
- Evaluates model performance using Mean Squared Error (MSE).
- Computes permutation feature importances to identify which features most affect the model's predictions.
- Aggregates feature importance scores by categories such as actor, producer, director, writer, and production company.

## Features

- Data preprocessing with pandas and NumPy
- Neural network implemented with PyTorch
- GPU acceleration if available
- Train/test split using scikit-learn
- Model evaluation with MSE metric
- Permutation-based feature importance calculation with progress bar (tqdm)
- Export of feature importance to CSV and aggregated results to text file

## Installation

Requires Python 3.x and the following packages:

```bash
pip install pandas numpy torch scikit-learn tqdm
