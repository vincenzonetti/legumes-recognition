from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.manifold import TSNE
from random_forest_train import plot_info


import random
import os
import joblib


def train_svm(X_train, y_train, kernel='linear', C=1, gamma='scale'):
    svm = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=C, gamma=gamma))
    svm.fit(X_train, y_train)
    return svm

#def grid_search_best_parameters(X_train,y_train):
#    C_range = [1e-2, 1, 1e2]
#    gamma_range = [1e-1, 1, 1e1]
#    kernel_range = ['linear', 'rbf', 'poly', 'sigmoid']
#    best_accuracy = 0
#    results = {}
#    for C in C_range:
#        for gamma in gamma_range:
#            for kernel in kernel_range:
#                svm = train_svm(X_train, y_train, C=C, gamma=gamma, kernel=kernel)
#                accuracy = svm.score(X_test, y_test)
#                results[(C, gamma, kernel)] = accuracy
#    #sort the results by accuracy
#    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
#    return sorted_results

def main():
    FILE = 'features.csv'
    df = pd.read_csv(FILE)
    feature_vectors = df.drop('label', axis=1).values
    labels = df['label'].values
    USE_PCA = False
    PCA_COMPONENTS = 5
    X = np.array(feature_vectors)  # Feature vectors
    y = np.array(labels)  # Labels for the seeds
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random.randint(0, 1000))
    kernel = 'rbf'
    if USE_PCA:
        pca = PCA(n_components=PCA_COMPONENTS)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    svm = train_svm(X_train, y_train,kernel=kernel)
    joblib.dump(svm, 'svm_model.pkl')
    accuracy = svm.score(X_test, y_test)
    print(f'Accuracy: {accuracy}')
    plot_info(X_train, y_train, y_test, X_test, svm)
        
        