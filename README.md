# legumes-recognition
SIV project for legumes recognition

# Project Overview
This project provides a simple workflow for:
1. Feature extraction from images containing a single object.
2. Training a classifier (SVM or Random Forest) using the extracted features.
3. Detecting and labeling multiple objects in new images, then producing statistics.

## Scripts and Usage

### 1) feature_extraction.py
• Extracts relevant features from seed images expected to contain exactly one object.  
• Saves all extracted features into a CSV file called `features.csv`.

### 2) svm_train.py or random_forest_train.py
• Uses the `features.csv` file to train a classification model.  
• You can choose either SVM or Random Forest Classifier according to your needs.  
• The trained model is saved in a file (e.g., `svm_model.pkl` or `random_forest_model.pkl`).

### 3) main.py
• Loads the trained model.  
• Reads images that may contain multiple objects and locates contours based on their area range.  
• Extracts features for each detected contour and predicts the label using the trained model.  
• Draws bounding boxes and labels on each detected object.  
• Displays statistics on how many objects of each label type are found in the image.

## Quick Steps:
1. Run `feature_extraction.py` to generate the `features.csv`.  
2. Train a model with either `svm_train.py` or `random_forest_train.py`.  
3. Run `main.py` to visualize detections, labels, and statistics for multiple objects in new images.
