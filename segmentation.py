import cv2 as cv
import numpy as np
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from feature_extraction import extract_contour_features
from tqdm.auto import tqdm

def remove_nested_bounding_boxes(contours):
    nested_contours = set()

    # Get bounding boxes for all contours
    bounding_boxes = [cv.boundingRect(cnt) for cnt in contours]

    for i, box1 in enumerate(bounding_boxes):
        x1, y1, w1, h1 = box1
        x1_end, y1_end = x1 + w1, y1 + h1

        for j, box2 in enumerate(bounding_boxes):
            if i != j:  # Avoid self-comparison
                x2, y2, w2, h2 = box2
                x2_end, y2_end = x2 + w2, y2 + h2

                # Check if box2 is completely inside box1
                if (
                    x2 >= x1 and y2 >= y1 and
                    x2_end <= x1_end and y2_end <= y1_end
                ):
                    nested_contours.add(j)  # Mark contour j as nested

    # Remove nested contours
    filtered_contours = [
        cnt for idx, cnt in enumerate(contours) if idx not in nested_contours
    ]

    return filtered_contours
    
def pipeline(difficoulty="Easy", df = None, classifier = None, SIZE = 1024, img = None):
    assert df is not None, 'Dataframe is required'
    assert classifier is not None, 'Classifier is required'
    assert img is not None, 'Image is required'

    img = cv.resize(img, (SIZE, SIZE))
        #crop by 100 px on top
    img = img[50:SIZE, 0:SIZE]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel = (9,9)
    if(difficoulty == 'Hard'):
        kernel = (15,15)
    blur = cv.GaussianBlur(gray, kernel, 0)

    thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(thresh, connectivity=8)
    
    # Filter treshold components
    for i in range(1, num_labels):  # Skip label 0 (background)
        area = stats[i, cv.CC_STAT_AREA]
        if area > 1000 or area < 100:  
            component_mask = (labels == i).astype(np.uint8)
            std_intensity = np.std(gray[component_mask == 1]) #calculate the standard deviation of the intensity
            if std_intensity < 10 or area < 10:thresh[component_mask == 1] = 0 #remove small or low information components

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=1)
 
    num_labels, labels = cv.connectedComponents(closed)

    # SEGMENTATION
    colored_labels = np.zeros((*labels.shape, 3), dtype=np.uint8)
    all_contours = []
    for label in range(1, num_labels):  # Skip label 0 (background)
        colored_labels[labels == label] = np.random.randint(0, 255, 3)
        component_mask = np.uint8(labels == label) * 255
        contours, _ = cv.findContours(component_mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        all_contours.extend(contours)

    #filter contours that are too big or too small
    all_contours = [contour for contour in all_contours if cv.contourArea(contour) > 40 and cv.contourArea(contour) < 5000]
    all_contours = remove_nested_bounding_boxes(all_contours)
    label_counter = {}
    bounding_box_img = img.copy()
    #FEATURE EXTRACTION and CLASSIFICATION
    for cnt in tqdm(all_contours, desc='Processing contours', position=0):
        features = extract_contour_features(img, cnt)
        feature_vector = [features[feature] for feature in df.columns[1:]]
        label = classifier.predict([feature_vector])[0]
        if label not in label_counter:
            label_counter[label] = 0
        label_counter[label] += 1
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(bounding_box_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(bounding_box_img, str(label), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return img,colored_labels,label_counter,bounding_box_img

if __name__ == '__main__':
    FOLDER = 'SIV_dataset/Test/'
    difficoulty = 'Easy'
    FOLDER = os.path.join(FOLDER, difficoulty)
    testFiles = os.listdir(FOLDER)
    # Load the model
    df = pd.read_csv('features.csv')
    
    svm = joblib.load('svm_model.pkl')
    random_forest = joblib.load('random_forest_model.pkl')
    SIZE = 1024
    for file in testFiles:
        img = cv.imread(os.path.join(FOLDER, file))
        img, colored_labels, label_counter, bounding_box_img = pipeline(difficoulty = difficoulty, df = df, classifier=random_forest, SIZE = SIZE, img = img)
        #make figure have the same height as the image
        fig = plt.figure(figsize=(10, 6))
        axs = fig.subplot_mosaic(
            [["img", "boxes"], ["segmentation", "historgram"]],
            gridspec_kw={"height_ratios": [2, 2]}  # Two rows: images get more space
        )
        
        axs['img'].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        axs['img'].set_title('Original image')
        axs['img'].axis('off')

        axs['boxes'].imshow(cv.cvtColor(bounding_box_img, cv.COLOR_BGR2RGB))
        axs['boxes'].set_title('Bounding box image')
        axs['boxes'].axis('off')

        axs['segmentation'].imshow(colored_labels)
        axs['segmentation'].set_title('Segmentation')
        axs['segmentation'].axis('off')

        axs['historgram'].bar(label_counter.keys(), label_counter.values())
        axs['historgram'].set_title('Seed distribution')
        plt.tight_layout()
        plt.show()
        
        
       
    
