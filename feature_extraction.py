import numpy as np
import cv2 as cv
import os
import pandas as pd
# Function to check if contour touches border
def touches_border(contour, img_width, img_height):
    x, y, w, h = cv.boundingRect(contour)
    return x == 0 or y == 0 or x + w >= img_width or y + h >= img_height

def extract_feature(img):
    #thresh = cv.adaptiveThreshold(imgGrey, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    contours = compute_contours(img)
    cnt = contours[0]
    area = cv.contourArea(cnt)
    perimeter = cv.arcLength(cnt, True)
    x, y, w, h = cv.boundingRect(cnt)
    aspect_ratio = w / h
    circularity = (4 * 3.1415 * area) / (perimeter ** 2)
    mask = np.zeros(img.shape, np.uint8)
    cv.drawContours(mask, [cnt], -1, 255, -1)
    mean_color = cv.mean(img, mask=mask)
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

def compute_crop_and_keypoints(img):
    img = cv.resize(img, (500, 500))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    #img = cv.drawKeypoints(img, keypoints, None)
    #crop the area around most important keypoint
    keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)
    x, y = keypoints[0].pt
    x, y = int(x), int(y)
    h, w = 100, 100
    crop = img[y-h:y+h, x-w:x+w]
    #key points and descriptors on the cropped image
    gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    img = cv.drawKeypoints(crop, keypoints, None)
    num_keypoints = len(keypoints)
    avg_keypoint_size = np.mean([keypoint.size for keypoint in keypoints])
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
    return crop,feature_dict

def compute_contours(img):
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    height, width = grey.shape

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(grey)
    
    treshold = cv.Canny(img_clahe,100,150)
    #check if all edges are 0
    if np.all(treshold == 0):
        treshold = cv.adaptiveThreshold(img_clahe, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    contours, hierarchy = cv.findContours(treshold, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    filtered_contours = [cnt for cnt in contours if not touches_border(cnt, width, height)]
    sorted_contours = sorted(filtered_contours, key=lambda c: cv.contourArea(cv.convexHull(c)), reverse=True)

    return sorted_contours
    
if __name__ == '__main__':
    
    df = pd.DataFrame(columns=['label','area', 'perimeter', 'aspect_ratio', 'circularity', 'mean_color_b', 'mean_color_g', 'mean_color_r'])
    
    PATH = 'SIV_dataset/'
    ## all images are in this folder, check them all
    subdirectories = os.listdir(PATH)
    #filter out files
    subdirectories = [subdir for subdir in subdirectories if os.path.isdir(os.path.join(PATH, subdir))]
    
    for subdir in subdirectories:
        if subdir == 'Test':
            continue
        images = os.listdir(os.path.join(PATH, subdir))
        images = [img for img in images if img.endswith('.jpg')]
        for filename in images:
            
            file_path = os.path.join(PATH, subdir, filename)
            img=cv.imread(file_path)
            
            crop, feature_dict = compute_crop_and_keypoints(img)
            feature_dict = {**feature_dict, **extract_feature(crop)}

            for key in feature_dict:
                feature_dict[key] = round(feature_dict[key], 2)
            new_row = {'label': subdir, **feature_dict}
            df = pd.concat([df, pd.DataFrame([new_row])])
    df.to_csv('features.csv', index=False)
