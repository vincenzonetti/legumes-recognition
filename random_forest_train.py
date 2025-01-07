from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
import os
import joblib
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay

def plot_confusion_matrix(y_true, X_test, svm,ax = None):
    if(ax is None):
        ax = plt.gca()
    y_pred = svm.predict(X_test)
    cm = confusion_matrix(y_true, y_pred)
    ax.matshow(cm, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    ax.set_xticks(range(len(set(y_true))), set(y_true))
    ax.set_yticks(range(len(set(y_true))), set(y_true))
    return ax

def TSNE_plot_data(X_train, y_train,ax = None):
    if(ax is None):
        ax = plt.gca()
    X_embedded = TSNE(n_components=2).fit_transform(X_train)
    labels = set(y_train)
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    
    for i, label in enumerate(labels):
        ax.scatter(X_embedded[y_train == label, 0], X_embedded[y_train == label, 1], color=colors[i], label=label)
    ax.legend()
    ax.axis('off')
    ax.set_title('TSNE')
    return ax

def PCA_plot_data(X_train, y_train,ax = None):
    if(ax is None):
        ax = plt.gca()
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    labels = set(y_train)
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    for i, label in enumerate(labels):
        ax.scatter(X_train[y_train == label, 0], X_train[y_train == label, 1], color=colors[i], label=label)
    ax.legend()
    ax.axis('off')
    ax.set_title('PCA')

def plot_decision_boundaries(X_train, y_train, svm,ax = None):
    if(ax is None):
        ax = plt.gca()
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    svm.fit(X_train, y_train)
    DecisionBoundaryDisplay.from_estimator(svm, X_train,ax=ax, xlabel='Decision Boundaries')
    
    return ax

def plot_info(X_train, y_train, y_true, X_test, svm):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plot_confusion_matrix(y_true, X_test, svm,axs[0, 0])
    TSNE_plot_data(X_train, y_train,axs[0, 1])
    PCA_plot_data(X_train, y_train,axs[1, 0])
    plot_decision_boundaries(X_train, y_train, svm,axs[1, 1])
    
    plt.show()
    
def forest_train(X_train, y_train):
    rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
    rf.fit(X_train, y_train)
    return rf

#load dataframe
def main():
    df = pd.read_csv('features.csv')
    feature_vectors = df.drop('label', axis=1).values
    labels = df['label'].values
    
    X = np.array(feature_vectors)  # Feature vectors
    y = np.array(labels)  # Labels for the seeds
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train Random Forest Classifier
    rf = forest_train(X_train, y_train)
    accuracy = rf.score(X_test, y_test)
    print(f'Accuracy: {accuracy}')
    joblib.dump(rf, 'random_forest_model.pkl')
    plot_info(X_train, y_train, y_test, X_test, rf)