import os
import glob
import yaml

# Define your dataset directories
train_dir = '../../huge_data/cards_kaggle_yolo/train'
val_dir = '../../huge_data/cards_kaggle_yolo/valid'

# Define your class names
class_names = [
    'ace of clubs', 'ace of diamonds', 'ace of hearts', 'ace of spades',
    'eight of clubs', 'eight of diamonds', 'eight of hearts', 'eight of spades',
    'five of clubs', 'five of diamonds', 'five of hearts', 'five of spades',
    'four of clubs', 'four of diamonds', 'four of hearts', 'four of spades',
    'jack of clubs', 'jack of diamonds', 'jack of hearts', 'jack of spades',
    'joker',
    'king of clubs', 'king of diamonds', 'king of hearts', 'king of spades',
    'nine of clubs', 'nine of diamonds', 'nine of hearts', 'nine of spades',
    'queen of clubs', 'queen of diamonds', 'queen of hearts', 'queen of spades',
    'seven of clubs', 'seven of diamonds', 'seven of hearts', 'seven of spades',
    'six of clubs', 'six of diamonds', 'six of hearts', 'six of spades',
    'ten of clubs', 'ten of diamonds', 'ten of hearts', 'ten of spades',
    'three of clubs', 'three of diamonds', 'three of hearts', 'three of spades',
    'two of clubs', 'two of diamonds', 'two of hearts', 'two of spades'
]


def convert_to_yolo_format(image_file):
    # Get the class label from the directory name
    class_label = class_names.index(os.path.basename(os.path.dirname(image_file)))

    # Since the bounding box is the entire image, the x_center and y_center are both 0.5 (halfway across),
    # and the width and height are both 1 (the full size of the image).
    # The resulting annotation is a string in the format 'class x_center y_center width height'
    yolo_annotation = f"{class_label} 0.5 0.5 1 1"

    return [yolo_annotation]


# Function to convert your annotations to YOLO format and save as .txt files
def convert_annotations(image_dir):
    for class_name in class_names:
        class_dir = os.path.join(image_dir, class_name)
        for image_file in glob.glob(os.path.join(class_dir, '*.jpg')):
            yolo_annotations = convert_to_yolo_format(image_file)

            # Save the YOLO annotations to a .txt file
            txt_file = os.path.splitext(image_file)[0] + '.txt'
            with open(txt_file, 'w') as f:
                f.write('\n'.join(yolo_annotations))


# Convert annotations for train and val datasets
convert_annotations(train_dir)
convert_annotations(val_dir)

# Create .yaml file
data = {
    'train': train_dir,
    'val': val_dir,
    'nc': len(class_names),
    'names': class_names
}

with open('../../huge_data/cards_kaggle_yolo/data.yaml', 'w') as f:
    yaml.dump(data, f)
