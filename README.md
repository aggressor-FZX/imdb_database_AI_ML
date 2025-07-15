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
- Permutation-based feature importance calculation with progress bar (`tqdm`)  
- Export of feature importance to CSV and aggregated results to text file  

## Installation

Requires Python 3.x and the following packages:

```bash
pip install pandas numpy torch scikit-learn tqdm
```

## Usage

Prepare your dataset CSV (`Movies_Parties_cat_BIG.csv`) in the working directory.

Run the training and evaluation script:

```bash
python your_script_name.py
```

Feature importance scores are saved to:

- `Permu_importance_df.csv` — detailed importances for all features  
- `employee_with_importance.csv` — original data with importance scores appended  
- `Aggregated_importanceNN.txt` — aggregated importance scores by category  

## Model Architecture

- Input layer with dimensionality equal to the number of features  
- Two hidden layers with 128 and 64 units respectively, using ReLU activation  
- Output layer producing a single continuous value for regression  

## Feature Importance Explained

Feature importance reveals which inputs your neural network relies on most when making predictions. Using **permutation importance**, we:
- Randomly shuffle the values of each feature one at a time
- Measure how much model performance drops (via MSE increase)
- Rank features by their impact

This is incredibly useful for:
- **Interpreting model decisions** in real-world terms (e.g. which actors or producers influence profit the most)
- **Reducing dimensionality** by eliminating low-impact features
- **Validating assumptions** about which aspects of the movie data matter most
- **Explaining results** to non-technical stakeholders

Aggregation by category helps identify broader trends — for example, if directors as a group are more predictive than writers or genres.

## Notes

- The code expects the dataset to have a `ProductionCompanies` column formatted as comma-separated strings  
- Adjust file paths and column names as necessary for your dataset  
- GPU will be used if available; otherwise CPU fallback occurs automatically  
- Permutation importance uses 2 repeats per feature by default for robustness  

