import numpy as np
import cv2 as cv
import os
import pandas as pd
from tqdm.auto import tqdm
# Function to check if contour touches border
def touches_border(contour, img_width, img_height):
    x, y, w, h = cv.boundingRect(contour)
    return x == 0 or y == 0 or x + w >= img_width or y + h >= img_height

def extract_contour_features(img=None,contour=None):
    cnt = contour
    area = cv.contourArea(cnt)
    perimeter = cv.arcLength(cnt, True)
    x, y, w, h = cv.boundingRect(cnt)
    aspect_ratio = w / h
    circularity = (4 * 3.1415 * area) / (perimeter ** 2)
    mask = np.zeros(img.shape[:2], np.uint8)
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


def compute_contours(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #binary treshold
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    dilated = cv.dilate(closed, kernel, iterations=2)
    eroded = cv.erode(dilated, kernel, iterations=1)
    edge = cv.Canny(eroded, 30, 150)

    contours, _ = cv.findContours(edge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    all_points = np.vstack(contours)
    hull = cv.convexHull(all_points)
    #img = cv.drawContours(img, [hull], -1, (0, 255, 0), 3)
    return hull

if __name__ == '__main__':
    
    df = pd.DataFrame(columns=['label','area', 'perimeter', 'aspect_ratio', 'circularity', 'mean_color_b', 'mean_color_g', 'mean_color_r'])
    
    PATH = 'SIV_dataset/Cropped/'  #<#
    ## all images are in this folder, check them all
    subdirectories = os.listdir(PATH)
    #filter out files
    subdirectories = [subdir for subdir in subdirectories if os.path.isdir(os.path.join(PATH, subdir))]
    
    for subdir in tqdm(subdirectories,desc='Processing Folder', position=0):
        images = os.listdir(os.path.join(PATH, subdir))
        images = [img for img in images if img.endswith('.jpg')]
        for filename in tqdm(images,desc='Processing Image', position=1):
            file_path = os.path.join(PATH, subdir, filename)
            img=cv.imread(file_path)
            img = cv.resize(img, (64, 64))
            contour = compute_contours(img.copy())
            feature_dict = extract_contour_features(img=img,contour=contour)
            #keypoint_features(img=combined)
            
            for key in feature_dict:
                feature_dict[key] = round(feature_dict[key], 2)
            new_row = {'label': subdir, **feature_dict}
            #check if df is empty
            if df.empty:
                df = pd.DataFrame([new_row])
            else: df = pd.concat([df, pd.DataFrame([new_row])])
    df.to_csv('features_cropped.csv', index=False)
