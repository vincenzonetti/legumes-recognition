from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
from feature_extraction import extract_feature
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
import os
import joblib

def plot_confusion_matrix(y_true, X_test, svm):
    y_pred = svm.predict(X_test)
    cm = confusion_matrix(y_true, y_pred)
    plt.matshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(range(len(set(y_true))), set(y_true))
    plt.yticks(range(len(set(y_true))), set(y_true))
    plt.show()

#load dataframe
if __name__ == '__main__':
    df = pd.read_csv('features.csv')
    feature_vectors = df.drop('label', axis=1).values
    labels = df['label'].values
    
    X = np.array(feature_vectors)  # Feature vectors
    y = np.array(labels)  # Labels for the seeds
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest Classifier
    rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
    rf.fit(X_train, y_train)
    
    accuracy = rf.score(X_test, y_test)
    print(f'Accuracy: {accuracy}')
    plot_confusion_matrix(y_test, X_test, rf)
    
    joblib.dump(rf, 'random_forest_model.pkl')