from constants import *

# Load train dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    str(DATASET_DIR / 'train'),
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

# Load validation dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    str(DATASET_DIR / 'valid'),
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

# Load test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    str(DATASET_DIR / 'test'),
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

# Get all class names
class_names = train_ds.class_names
