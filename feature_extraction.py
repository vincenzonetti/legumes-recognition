import numpy as np
import cv2 as cv
import os
import pandas as pd
from tqdm.auto import tqdm
from skimage.measure import shannon_entropy
from scipy.signal import correlate2d
from skimage.feature import graycomatrix, graycoprops
# Function to check if contour touches border
def touches_border(contour, img_width, img_height):
    x, y, w, h = cv.boundingRect(contour)
    return x == 0 or y == 0 or x + w >= img_width or y + h >= img_height

def compute_autocorrelation(gray_img):
    autocorr = correlate2d(gray_img, gray_img, mode='full')
    central_region = autocorr[gray_img.shape[0]//2-5:gray_img.shape[0]//2+5,
                              gray_img.shape[1]//2-5:gray_img.shape[1]//2+5]
    return np.mean(central_region), np.std(central_region)

def compute_fractal_dimension(binary_img):
    # Box-counting method for fractal dimension
    sizes = np.arange(2, 10)
    counts = [np.sum(binary_img[::s, ::s]) for s in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return coeffs[0]



def compute_glcm_features(gray_img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    glcm = graycomatrix(gray_img, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    energy = graycoprops(glcm, 'energy').mean()
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))  # Add small value to avoid log(0)
    return contrast, correlation, energy, entropy



def extract_contour_features(img=None,contour=None):
    img_height, img_width = img.shape[:2]
    cnt = contour
    area = cv.contourArea(cnt)
    perimeter = cv.arcLength(cnt, True)
    x, y, w, h = cv.boundingRect(cnt)
    circularity = (4 * 3.1415 * area) / (perimeter ** 2)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv.drawContours(mask, [cnt], -1, 255, -1)
    mean_color = cv.mean(img, mask=mask)
    (x,y), radius = cv.minEnclosingCircle(cnt)
    
    area = cv.contourArea(cnt)
    perimeter = cv.arcLength(cnt, True)
    #area and perimeter confuse the classifier because they depend on how much we zoom in the image thus they will be not used
    #show this in the report
    mean_stddev = cv.meanStdDev(img, mask=mask)
    color_std_b = mean_stddev[1][0][0]
    color_std_g = mean_stddev[1][1][0]
    color_std_r = mean_stddev[1][2][0]
    entropy = shannon_entropy(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #autocorrelation_mean, autocorrelation_std = compute_autocorrelation(gray)
    fractal_dimension = compute_fractal_dimension(gray)
    contrast, correlation, _, entropy_glcm = compute_glcm_features(gray)
    normalized_radius = radius / min(w, h) 
    normalized_area = area / (img_width * img_height)
    normalized_perimeter = perimeter / (img_width + img_height)
    median_color_b = np.median(img[..., 0][mask == 255])
    median_color_g = np.median(img[..., 1][mask == 255])
    median_color_r = np.median(img[..., 2][mask == 255])
    hist_b = cv.calcHist([img], [0], mask, [256], [0, 256]).flatten()
    hist_g = cv.calcHist([img], [1], mask, [256], [0, 256]).flatten()
    hist_r = cv.calcHist([img], [2], mask, [256], [0, 256]).flatten()

    feature_dict = {
        'circularity': circularity,
        'mean_color_b': mean_color[0],
        'mean_color_g': mean_color[1],
        'mean_color_r': mean_color[2],
        'radius': radius,
        'extent': area / (w * h),
        'normalized_radius': normalized_radius,
        'normalized_area': normalized_area,
        'normalized_perimeter': normalized_perimeter,
        'median_color_b': median_color_b,
        'median_color_g': median_color_g,
        'median_color_r': median_color_r,
        'entropy': entropy,
        'fractal_dimension': fractal_dimension,
        'contrast': contrast,
        'correlation': correlation,
        'entropy_glcm': entropy_glcm
        }
    for i,hist in enumerate([hist_b, hist_g, hist_r]):
        for j in range(256):
            feature_dict[f'hist_{i}_{j}'] = hist[j]
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
    df.to_csv('features.csv', index=False)
