# ISIC 2024 Challenge - Skin Lesion Classification

# Overview
This repository contains a solution for the ISIC 2024 Challenge, focusing on skin lesion classification using a blend of advanced machine learning algorithms. The model leverages LightGBM, CatBoost, and XGBoost classifiers to achieve competitive performance in predicting skin lesion malignancy.

# Objective
To classify skin lesions into malignant or benign categories using a dataset of skin lesion images, with a goal to achieve high predictive accuracy on the Kaggle leaderboard.

# Data
Training Data: train-metadata.csv
Test Data: test-metadata.csv
Submission Sample: sample_submission.csv

# Libraries and Dependencies

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, log_evaluation
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

# Configuration
class Config():
    seed = 2024
    num_folds = 10
    TARGET_NAME = 'target'

# Data Preparation
Loading Data: Import the training and test datasets.
Feature Engineering: Create new features based on existing columns to improve model performance.
Feature Selection: Remove highly correlated features to reduce redundancy.

# Feature Engineering
Features include:

Anatomical site and lesion characteristics
Image attributes and color metrics
Age and copyright license-based features    

# Metrics
The model is evaluated using the partial AUC (pAUC) metric, specifically targeting a minimum true positive rate (TPR) of 0.8.
def pauc_above_tpr(y_true, y_pred):
    min_tpr = 0.8
    v_gt = abs(np.asarray(y_true) - 1)
    v_pred = np.array([1.0 - x for x in y_pred])
    max_fpr = abs(1 - min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    return 'pauc', partial_auc, True
 # Model Training
Three models are trained and evaluated:
LightGBM
CatBoost
XGBoost

Model parameters and training details are specified in the code. Each model is validated using Stratified Group K-Fold cross-validation.

# Results
The model achieved a leaderboard score of 0.164. Predictions are made by averaging the outputs of the three models.

