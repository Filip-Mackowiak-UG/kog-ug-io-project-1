import tensorflow as tf
import pathlib

# GLOBAL CONSTANTS
AUTOTUNE = tf.data.AUTOTUNE
DATASET_DIR = pathlib.Path('huge_data/cards_kaggle')

BATCH_SIZE = 16
IMG_HEIGHT = 128
IMG_WIDTH = 128
