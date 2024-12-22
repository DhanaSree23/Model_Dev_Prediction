import streamlit as st
import pandas as pd
from main import main
from sklearn.model_selection import train_test_split

def read_and_split_data(file_path):
    # Read data
    data = pd.read_csv(file_path)
    # Assuming the last column is the target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

st.title("Model Evaluation Metrics")

file_path = st.text_input("Enter the path to your data file")

if file_path:
    X_train, X_test, y_train, y_test = read_and_split_data(file_path)
    
    # Train models and save them as pickle files
    from main import ModelTraining
    model_training = ModelTraining(X_train, y_train)
    model_training.train_logistic_regression()
    model_training.train_decision_tree()
    model_training.train_random_forest()
    model_training.train_svm()
    model_training.train_naive_bayes()

    # Evaluate the models and display the metrics
    metrics_df = main(X_test, y_test)
    st.write(metrics_df)
