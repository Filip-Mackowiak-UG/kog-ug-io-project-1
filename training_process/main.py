from pathlib import Path
from datasets import *
from utils import save_accuracy, save_example_images_from_provided, show_img_count
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

from experiments.model_8.model import define_model, augment_data

charts_path = Path("other/charts")


def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def add_padding(image, label):
    original_batch_size, original_height, original_width, _ = tf.unstack(tf.shape(image))
    top_padding = 20
    left_padding = 20

    # Resize the image to a smaller size so input and output size stay the same
    smaller_height = original_height - 2 * top_padding
    smaller_width = original_width - 2 * left_padding
    image = tf.image.resize(image, [smaller_height, smaller_width])

    # Create paddings
    paddings = tf.constant([[0, 0], [top_padding, top_padding], [left_padding, left_padding], [0, 0]])

    # Pad the image with a colour
    image = tf.pad(image, paddings, "CONSTANT", constant_values=1.0)

    return image, label


augmentation_model = augment_data()


def apply_augmentation(image, label):
    augmented_image = augmentation_model(image, training=True)
    return augmented_image, label


show_img_count()

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Define learning rate schedule
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

train_img_count = len(list(DATASET_DIR.glob('train/*/*.jpg')))
valid_img_count = len(list(DATASET_DIR.glob('valid/*/*.jpg')))

# Normalise data
train_ds = train_ds.map(normalize)
val_ds = val_ds.map(normalize)

# Save example photos without padding
save_example_images_from_provided(train_ds, str(charts_path / "example_images_without_padding.jpg"))

# train_ds = train_ds.map(add_padding)

# Save example photos with padding
save_example_images_from_provided(train_ds, str(charts_path / "example_images_with_padding.jpg"))

train_ds = train_ds.map(apply_augmentation)

# Save example photos with padding, and after augmentation
save_example_images_from_provided(train_ds, str(charts_path / "augmented.jpg"))

train_ds = train_ds.cache().shuffle(train_img_count).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().shuffle(valid_img_count).prefetch(buffer_size=AUTOTUNE)

model = define_model()

model.summary()

epochs = 120
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=epochs
# )

cp_callback_acc = ModelCheckpoint(filepath='model_acc.{epoch:02d}-{val_accuracy:.2f}.keras',
                                  monitor='val_accuracy',
                                  verbose=1,
                                  save_best_only=True,
                                  mode='max')

cp_callback_loss = ModelCheckpoint(filepath='model_loss.{epoch:02d}-{val_loss:.2f}.keras',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   mode='auto')

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stopping, lr_schedule]
)

n_epochs = len(history.history['accuracy'])

save_accuracy(history, n_epochs)
model.save("kog.keras")
