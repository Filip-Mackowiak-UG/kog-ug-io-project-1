import os
import shutil

# Define your dataset directories
# root_dir = './huge_data/cards_kaggle_yolo/valid_pre'
# output_dir = './huge_data/cards_kaggle_yolo/valid'
root_dir = './huge_data/cards_kaggle_yolo/train_pre'
output_dir = '../../huge_data/cards_kaggle_yolo/train'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

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

# For each class name
for class_name in class_names:
    # Get the directory for this class
    class_dir = os.path.join(root_dir, class_name)

    # For each file in this directory
    for filename in os.listdir(class_dir):
        # Get the full path to the file
        old_path = os.path.join(class_dir, filename)

        # Create a new filename that includes the class name
        new_filename = f"{class_name.replace(' ', '_')}_{filename}"

        # Get the full path to the new file
        new_path = os.path.join(output_dir, new_filename)

        # Move the file
        shutil.move(old_path, new_path)
