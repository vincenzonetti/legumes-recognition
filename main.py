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
    avg_keypoint_size = np.mean([keypoint.size for keypoint in keypoints])
    if(num_keypoints == 0): avg_keypoint_size = 0
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

def compute_informations(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    binary_img = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #sort them by area and filter out the ones that are too small or too big
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area and cv2.contourArea(cnt) <= max_area]
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
        img = cv2.resize(img, (800, 600))
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
        
       
    
