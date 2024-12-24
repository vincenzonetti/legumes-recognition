import cv2
import numpy as np
import os
import pandas as pd
import joblib

if __name__ == '__main__':
    # Load the model
    df = pd.read_csv('features.csv')
    max_area = df['area'].max()
    min_area = df['area'].min()
    svm = joblib.load('svm_model.pkl')
    
    # Load the image
    folder_name = 'dataset'
    file_name = 'test.jpg'
    img = cv2.imread(os.path.join(folder_name, file_name))
    #resize image
    img = cv2.resize(img, (800, 600))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #filter out contours with area outside the range
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area and cv2.contourArea(cnt) <= max_area]
    #visualize contours
    seed_features = []
    for cnt in filtered_contours:
    # Compute features for each seed
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h
        circularity = (4 * 3.1415 * area) / (perimeter ** 2)
        mean_color = cv2.mean(img, mask=cv2.drawContours(np.zeros_like(gray), [cnt], -1, 255, -1))

        # Combine into a feature vector
        feature_vector = [area, perimeter, aspect_ratio, circularity, mean_color[0], mean_color[1], mean_color[2]]
        seed_features.append(feature_vector)
    # Use the trained SVM to predict labels for each seed
    predicted_labels = svm.predict(seed_features)

    # Annotate the image with predictions
    for i, cnt in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        label = predicted_labels[i]

        # Draw bounding box and label on the original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv2.imshow('Labeled Seeds', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    
