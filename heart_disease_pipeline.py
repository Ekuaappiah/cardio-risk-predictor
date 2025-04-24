#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Configuration
FILE_PATH = 'data/heart.csv'
MODELS_DIR = Path('models')
MODEL_PATH = MODELS_DIR / 'heart_disease_predictor.pkl'


# Data Handling
def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError("Dataset file not found.")


def split_data(df):
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


# Model Training and Tuning
def initial_train(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def tune_model(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'class_weight': [None, 'balanced']
    }

    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_weighted',
        'recall': 'recall_weighted',
        'f1': 'f1_weighted'
    }

    model = RandomForestClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit='f1',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


# Evaluation
def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)
    return acc, report


def cross_validate(model, X_train, y_train):
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return scores


# Save/Load Model
def save_model(model):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Train the model first.")
    return joblib.load(MODEL_PATH)


# Inference
def predict(model, input_dict):
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)
    probabilities = model.predict_proba(input_df)
    return prediction[0], probabilities[0][1]


# Main Execution

def main():
    df = load_data(FILE_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    base_model = initial_train(X_train, y_train)
    best_model = tune_model(X_train, y_train)

    val_acc, val_report = evaluate_model(best_model, X_val, y_val)
    print(f"Validation Accuracy: {val_acc}")
    print("\nClassification Report:\n", val_report)

    save_model(best_model)


if __name__ == "__main__":
    main()