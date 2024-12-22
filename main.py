import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='macro'),
        'Recall': recall_score(y_test, y_pred, average='macro'),
        'F1 Score': f1_score(y_test, y_pred, average='macro')
    }
    return metrics

def main(X_test, y_test):
    models = ['logistic_regression.pkl', 'decision_tree.pkl', 'random_forest.pkl', 'svm.pkl', 'naive_bayes.pkl']
    metrics_list = []

    for model_file in models:
        model = load_model(model_file)
        metrics = evaluate_model(model, X_test, y_test)
        metrics['Model'] = model_file.split('.')[0]
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df

class ModelTraining:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def train_logistic_regression(self):
        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)
        with open('logistic_regression.pkl', 'wb') as file:
            pickle.dump(model, file)

    def train_decision_tree(self):
        model = DecisionTreeClassifier()
        model.fit(self.X_train, self.y_train)
        with open('decision_tree.pkl', 'wb') as file:
            pickle.dump(model, file)

    def train_random_forest(self):
        model = RandomForestClassifier()
        model.fit(self.X_train, self.y_train)
        with open('random_forest.pkl', 'wb') as file:
            pickle.dump(model, file)

    def train_svm(self):
        model = SVC()
        model.fit(self.X_train, self.y_train)
        with open('svm.pkl', 'wb') as file:
            pickle.dump(model, file)

    def train_naive_bayes(self):
        model = GaussianNB()
        model.fit(self.X_train, self.y_train)
        with open('naive_bayes.pkl', 'wb') as file:
            pickle.dump(model, file)
