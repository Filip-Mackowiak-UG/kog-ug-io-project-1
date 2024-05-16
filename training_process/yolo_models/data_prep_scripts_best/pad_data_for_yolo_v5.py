import cv2
import os
import random
import numpy as np


def adjust_bboxes(bboxes, pad_x, pad_y, img_size, new_size):
    adjusted_bboxes = []
    for bbox in bboxes:
        class_id, x, y, w, h = map(float, bbox.split())
        x = (x * img_size[1] + pad_x) / new_size[1]
        y = (y * img_size[0] + pad_y) / new_size[0]
        w = w * img_size[1] / new_size[1]
        h = h * img_size[0] / new_size[0]
        adjusted_bboxes.append(f"{int(class_id)} {x} {y} {w} {h}")
    return adjusted_bboxes


directory = "./huge_data/cards_kaggle_yolo_padded_v5/train"

for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        img_path = os.path.join(directory, filename)
        txt_path = os.path.join(directory, filename.replace(".jpg", ".txt"))

        # Load image
        img = cv2.imread(img_path)

        # Add padding
        pad_x = random.randint(5, img.shape[1])  # Random padding for width
        pad_y = random.randint(5, img.shape[0])  # Random padding for height

        # Create a random-colored background image
        bg = np.random.randint(0, 256, (img.shape[0] + 2 * pad_y, img.shape[1] + 2 * pad_x, 3), dtype=np.uint8)

        # Place the original image in the center of the background image
        bg[pad_y:-pad_y, pad_x:-pad_x] = img

        # Save the padded image, overwriting the original image
        cv2.imwrite(img_path, bg)

        # Adjust bounding boxes in .txt file
        with open(txt_path, "r") as f:
            bboxes = f.readlines()
        bboxes = adjust_bboxes(bboxes, pad_x, pad_y, img.shape, bg.shape)
        with open(txt_path, "w") as f:
            f.write("\n".join(bboxes))
