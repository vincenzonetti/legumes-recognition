from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
from feature_extraction import extract_feature
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.manifold import TSNE



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

def train_svm(X_train, y_train, kernel='linear'):
    svm = make_pipeline(StandardScaler(), SVC(kernel=kernel))
    svm.fit(X_train, y_train)
    return svm

def TSNE_plot_data(X_train, y_train):
    X_embedded = TSNE(n_components=2).fit_transform(X_train)
    labels = set(y_train)
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    for i, label in enumerate(labels):
        plt.scatter(X_embedded[y_train == label, 0], X_embedded[y_train == label, 1], color=colors[i], label=label)
    plt.legend()
    plt.axis('off')
    plt.title('TSNE plot of training data')
    plt.show()

def PCA_plot_data(X_train, y_train):
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    labels = set(y_train)
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    for i, label in enumerate(labels):
        plt.scatter(X_train[y_train == label, 0], X_train[y_train == label, 1], color=colors[i], label=label)
    plt.legend()
    plt.axis('off')
    plt.title('PCA plot of training data')
    plt.show()

def plot_decision_boundaries(X_train, y_train, svm):
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    svm.fit(X_train, y_train)
    DecisionBoundaryDisplay.from_estimator(svm, X_train)
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('features.csv')
    feature_vectors = df.drop('label', axis=1).values
    labels = df['label'].values
    USE_PCA = True
    PCA_COMPONENTS = 5
    X = np.array(feature_vectors)  # Feature vectors
    y = np.array(labels)  # Labels for the seeds
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    kernel = 'linear'
    if USE_PCA:
        pca = PCA(n_components=PCA_COMPONENTS)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    
    svm = train_svm(X_train, y_train,kernel=kernel)
    #svm = joblib.load('svm_model.pkl')
    accuracy = svm.score(X_test, y_test)
    print(f'Accuracy: {accuracy}')
    plot_confusion_matrix(y_test, X_test, svm)
    #TSNE_plot_data(X_train,y_train)
    #PCA_plot_data(X_train,y_train)
    plot_decision_boundaries(X_train, y_train, svm)
    
    joblib.dump(svm, 'svm_model.pkl')