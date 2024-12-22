import numpy as np
import cv2 as cv
import os
import pandas as pd
# Function to check if contour touches border
def touches_border(contour, img_width, img_height):
    x, y, w, h = cv.boundingRect(contour)
    return x == 0 or y == 0 or x + w >= img_width or y + h >= img_height

def extract_feature(image_path):
    img = cv.imread(image_path)
    #resize image
    img = cv.resize(img, (500, 500))
    imgGrey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    height, width = imgGrey.shape
    ret, thresh = cv.threshold(imgGrey, 127, 255, 0)
    # Change to RETR_CCOMP to get internal contours
    contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    # Filter out contours that touch the border
    filtered_contours = [cnt for cnt in contours if not touches_border(cnt, width, height)]

    sorted_contours = sorted(filtered_contours, key=lambda c: cv.contourArea(cv.convexHull(c)), reverse=True)

    #keep only the first 5 contours
    sorted_contours = sorted_contours[:1]

    #for i, contour in enumerate(sorted_contours):
    #    cv.drawContours(img, [contour], -1, (0,255,0), 3)
    #    cv.imshow(f'Contour {i}', img)
    #    cv.waitKey(0)
    #
    #cv.destroyAllWindows()

    cnt = sorted_contours[0]
    area = cv.contourArea(cnt)
    perimeter = cv.arcLength(cnt, True)
    x, y, w, h = cv.boundingRect(cnt)
    aspect_ratio = w / h
    circularity = (4 * 3.1415 * area) / (perimeter ** 2)

    mask = np.zeros(imgGrey.shape, np.uint8)
    cv.drawContours(mask, [cnt], -1, 255, -1)
    pixelpoints = cv.findNonZero(mask)
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

if __name__ == '__main__':
    
    df = pd.DataFrame(columns=['label','area', 'perimeter', 'aspect_ratio', 'circularity', 'mean_color_b', 'mean_color_g', 'mean_color_r'])
    
    folder_path = 'dataset/'
    ## all images are in this folder, check them all
    subdirectories = os.listdir(folder_path)
    #filter out files
    subdirectories = [subdir for subdir in subdirectories if os.path.isdir(os.path.join(folder_path, subdir))]
    breakpoint()
    for subdir in subdirectories:
        images = os.listdir(os.path.join(folder_path, subdir))
        images = [img for img in images if img.endswith('.jpg')]
        for filename in images:
            file_path = os.path.join(folder_path, subdir, filename)
            feature_dict = extract_feature(file_path)
            #rounding off the values
            for key in feature_dict:
                feature_dict[key] = round(feature_dict[key], 2)
            new_row = {'label': subdir, **feature_dict}
            df = pd.concat([df, pd.DataFrame([new_row])])
    df.to_csv('features.csv', index=False)
