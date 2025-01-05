import cv2 as cv
import numpy as np
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt


    
if __name__ == '__main__':
    FOLDER = 'SIV_dataset/Test/Easy'
    testFiles = os.listdir(FOLDER)
    # Load the model
    df = pd.read_csv('features.csv')
    
    svm = joblib.load('svm_model.pkl')
    
    for file in testFiles:
        img = cv.imread(os.path.join(FOLDER, file))
        img = cv.resize(img, (480, 480))
        #crop by 100 px on top
        img = img[50:480, 0:480]
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        

        #_, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        #adaptive threshold
        thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(thresh, connectivity=8)
    
        # Analyze each component
        for i in range(1, num_labels):  # Skip label 0 (background)
            # Get component size
            area = stats[i, cv.CC_STAT_AREA]
            
            # If component is large (adjust threshold as needed)
            
            if area > 1000 or area < 100:  
                
            
                component_mask = (labels == i).astype(np.uint8)
                
                # Calculate average intensity in original grayscale image
                std_intensity = np.std(gray[component_mask == 1])
                # If component is bright, mark as background
                if std_intensity < 10 or area < 10:  # Adjust threshold as needed
                    thresh[component_mask == 1] = 0
        
        #closing
       
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=1)
 
        # Find connected components
        num_labels, labels = cv.connectedComponents(closed)

        # Create a random color map for visualization
        colored_labels = np.zeros((*labels.shape, 3), dtype=np.uint8)
        for label in range(1, num_labels):  # Skip label 0 (background)
            colored_labels[labels == label] = np.random.randint(0, 255, 3)

        # Display results
        cv.imshow('Original Mask', closed)
        cv.imshow('Segmented Objects (Connected Components)', colored_labels)
        cv.waitKey(0)
        cv.destroyAllWindows()


        edges = cv.Canny(closed, 30, 150)
        contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        #filter contours that are too big or too small
        contours = [cnt for cnt in contours if cv.contourArea(cnt) > 50 and cv.contourArea(cnt) < 2000]
        cv.drawContours(img, contours, -1, (0, 255, 0), 2)

     
        cv.imshow('img', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        
        

        

        
        
        #make figure have the same height as the image
        fig = plt.figure(figsize=(10, 6))
        axs = fig.subplot_mosaic(
            [["img", "bounding"], ["historgram", "historgram"]],
            gridspec_kw={"height_ratios": [2, 1]}  # Two rows: images get more space
        )
        """
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
        """
        
       
    
