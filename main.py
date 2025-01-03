import cv2
import numpy as np
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

def compute_keypoints_features(img_crop):
    #img_crop = cv2.resize(img_crop, (50, 50))
    h,w = img_crop.shape[:2]
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)
    try:
        x, y = keypoints[0].pt
        x, y = int(x), int(y)
    except IndexError:
        x, y = h//2, w//2
    num_keypoints = len(keypoints)
    
    if(num_keypoints == 0): avg_keypoint_size = 0
    else: avg_keypoint_size = np.mean([keypoint.size for keypoint in keypoints]) 
    if(descriptors is None):
        descriptor_mean = 0
        descriptor_std = 0
    else:
        descriptor_mean = np.mean(descriptors)
        descriptor_std = np.std(descriptors)

    feature_dict = {
        'num_keypoints': num_keypoints,
        'avg_keypoint_size': avg_keypoint_size,
        'descriptor_mean': descriptor_mean,
        'descriptor_std': descriptor_std
    }
    return feature_dict

def extract_contour_features(img,contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h
    circularity = (4 * 3.1415 * area) / (perimeter ** 2)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(grey.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean_color = cv2.mean(img, mask=mask)
    
    
    feature_dict = {
        'area': area,
        'perimeter': perimeter,
        'aspect_ratio': aspect_ratio,
        'circularity': circularity,
        'mean_color_b': mean_color[0],
        'mean_color_g': mean_color[1],
        'mean_color_r': mean_color[2]
    }
    return feature_dict

def remove_nested_bounding_boxes(contours):
    nested_contours = set()

    # Get bounding boxes for all contours
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

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

def get_contours(img=None, min_area=100, max_area=1000):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    blurred = cv2.bilateralFilter(gray, d=5, sigmaColor=20, sigmaSpace=30)
    img_clahe = clahe.apply(blurred)
    
    binary_img = cv2.Canny(img_clahe,100,150)
    #binary_img = cv2.adaptiveThreshold(img_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            filtered_contours.append(cnt)  # Keep contours within area range
        elif area > max_area:
            # Reapply contouring for larger areas using cv2.RETR_LIST
            x, y, w, h = cv2.boundingRect(cnt)
            region_of_interest = binary_img[y:y+h, x:x+w]  # Extract region of interest
            
            # Second pass contouring
            secondary_contours, _ = cv2.findContours(
                region_of_interest, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            secondary_contours = [scnt for scnt in secondary_contours if cv2.contourArea(scnt) <= max_area and cv2.contourArea(scnt) >= min_area]
            # Adjust coordinates back to the original image
            for scnt in secondary_contours:
                scnt[:, 0, 0] += x  # Adjust x-coordinates
                scnt[:, 0, 1] += y  # Adjust y-coordinates
                filtered_contours.append(scnt)
    filtered_contours = remove_nested_bounding_boxes(filtered_contours)  
    return filtered_contours

def compute_informations(img,min_area=100,max_area=1000,svm=None,df=None):
    filtered_contours = get_contours(img=img, min_area=min_area, max_area=max_area)
    label_counter = {}
    for cnt in filtered_contours:
        x,y, w, h = cv2.boundingRect(cnt)
        seed_crop = img[y:y+h, x:x+w]
        keypoints_feature = compute_keypoints_features(seed_crop)
        contour_feature = extract_contour_features(img, cnt)
        feature_dict = {**keypoints_feature, **contour_feature}
        feature_vector = [feature_dict[feature] for feature in df.columns[1:]]
        label = svm.predict([feature_vector])[0]
        if label not in label_counter:
            label_counter[label] = 0
        label_counter[label] += 1
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, str(label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    bounding_box_img = img
    return bounding_box_img, label_counter

    
if __name__ == '__main__':
    FOLDER = 'SIV_dataset/Test'
    testFiles = os.listdir(FOLDER)
    # Load the model
    df = pd.read_csv('features.csv')
    max_area = df['area'].max()
    min_area = df['area'].min()
    svm = joblib.load('svm_model.pkl')
    
    for file in testFiles:
        img = cv2.imread(os.path.join(FOLDER, file))
        img = cv2.resize(img, (500, 500))
        #crop image pixels on borders by 10%
        img = img[20:-20, 20:-20]
        bounding_box_img, label_counter =compute_informations(img.copy())
        #make figure have the same height as the image
        fig = plt.figure(figsize=(10, 6))
        axs = fig.subplot_mosaic(
            [["img", "bounding"], ["historgram", "historgram"]],
            gridspec_kw={"height_ratios": [2, 1]}  # Two rows: images get more space
        )
        axs['img'].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs['img'].set_title('Original image')
        axs['img'].axis('off')

        axs['bounding'].imshow(cv2.cvtColor(bounding_box_img, cv2.COLOR_BGR2RGB))
        axs['bounding'].set_title('Bounding box image')
        axs['bounding'].axis('off')

        
        axs['historgram'].bar(label_counter.keys(), label_counter.values())
        axs['historgram'].set_title('Seed distribution')
        plt.tight_layout()
        plt.show()
        
       
    
