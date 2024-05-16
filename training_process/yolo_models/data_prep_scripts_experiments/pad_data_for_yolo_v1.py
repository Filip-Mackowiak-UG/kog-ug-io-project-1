import cv2
import os
import random


def adjust_bboxes(bboxes, pad_x, pad_y, img_size):
    adjusted_bboxes = []
    for bbox in bboxes:
        class_id, x, y, w, h = map(float, bbox.split())
        x = (x * img_size + pad_x) / (img_size + 2 * pad_x)
        y = (y * img_size + pad_y) / (img_size + 2 * pad_y)
        w = w * img_size / (img_size + 2 * pad_x)
        h = h * img_size / (img_size + 2 * pad_y)
        adjusted_bboxes.append(f"{int(class_id)} {x} {y} {w} {h}")
    return adjusted_bboxes


# Directory containing your images and .txt files
# directory = "./huge_data/cards_kaggle_yolo_padded/valid"
directory = "./huge_data/cards_kaggle_yolo_padded/train"

for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        img_path = os.path.join(directory, filename)
        txt_path = os.path.join(directory, filename.replace(".jpg", ".txt"))

        # Load image
        img = cv2.imread(img_path)

        # Add random padding
        pad_x = random.randint(5, 50)
        pad_y = random.randint(5, 50)
        img = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Resize image back to original size
        img = cv2.resize(img, (224, 224))

        # Save the padded image, overwriting the original image
        cv2.imwrite(img_path, img)

        # Adjust bounding boxes in .txt file
        with open(txt_path, "r") as f:
            bboxes = f.readlines()
        bboxes = adjust_bboxes(bboxes, pad_x, pad_y, 224)
        with open(txt_path, "w") as f:
            f.write("\n".join(bboxes))
