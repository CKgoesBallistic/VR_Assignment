import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return image, gray, blur

def detect_coins(image, blur, min_area=1200):
    canny = cv2.Canny(blur, 20, 150)
    dilated = cv2.dilate(canny, None, iterations=2)
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter small contours based on area
    filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    
    segmented = image.copy()
    
    for idx, contour in enumerate(filtered_contours):
        (x, y), radius = cv2.minEnclosingCircle(contour)  # Fit minimum enclosing circle
        center = (int(x), int(y))
        radius = int(radius)

        color = [random.randint(0, 255) for _ in range(3)]
        cv2.circle(segmented, center, radius, color, thickness=-1)  # Filled circle
        cv2.putText(segmented, str(idx+1), (center[0]-10, center[1]+10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    num_coins = len(filtered_contours)
    print(f"Final Coin Count: {num_coins}")

    return segmented, num_coins

def main(image_path):
    image, gray, blur = preprocess_image(image_path)
    
    segmented, num_coins = detect_coins(image, blur)

    plt.figure(figsize=(12,6))
    
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    plt.title(f'Segmented Coins ({num_coins} detected)')
    plt.axis('off')
    
    plt.show()

# Run the function with the new uploaded image
image_path = "input/Part1_input_image_2.jpg"
main(image_path)
