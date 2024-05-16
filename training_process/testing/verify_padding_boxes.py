import cv2
import numpy as np
import os

# Load image
img_path = './huge_data/cards_kaggle_yolo_padded_v5/valid/ace_of_clubs_4.jpg'
img = cv2.imread(img_path)

if img is None:
    print(f"Could not open or find the image: {img_path}")
else:
    # Load bounding box values from txt file
    txt_path = img_path.replace(".jpg", ".txt")
    with open(txt_path, 'r') as f:
        class_id, x, y, w, h = map(float, f.readline().split())

    # Convert normalized coordinates to pixel values
    x, y, w, h = x * img.shape[1], y * img.shape[0], w * img.shape[1], h * img.shape[0]

    # Calculate top-left and bottom-right coordinates
    x1, y1 = int(x - w/2), int(y - h/2)
    x2, y2 = int(x + w/2), int(y + h/2)

    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display image
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
