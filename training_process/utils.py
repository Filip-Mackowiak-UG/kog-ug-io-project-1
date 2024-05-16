import matplotlib.pyplot as plt
from datasets import *
import numpy as np


def denormalize(image):
    return tf.cast(image * 255, tf.uint8)


def show_img_count():
    image_count1 = len(list(DATASET_DIR.glob('test/*/*.jpg')))
    print(f"test: {image_count1}")
    image_count2 = len(list(DATASET_DIR.glob('train/*/*.jpg')))
    print(f"train: {image_count2}")
    image_count3 = len(list(DATASET_DIR.glob('valid/*/*.jpg')))
    print(f"valid: {image_count3}")
    image_count4 = len(list(DATASET_DIR.glob('*/*/*.jpg')))
    print(image_count4)
    print(f"Sum: {image_count1 + image_count2 + image_count3}")

def save_example_images_from_provided(dataset, filename, augmentation_layer=None):
    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_facecolor('grey')
    for images, labels in dataset.take(1):
        if augmentation_layer is not None:
            # Check the rank of the images tensor
            if len(images.shape) == 4:
                images = augmentation_layer(images)
            else:
                print(f"Unexpected images tensor shape: {images.shape}, expected rank 4.")
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            if tf.reduce_min(images[i]) >= 0 and tf.reduce_max(images[i]) <= 1:  # Check if the image is normalized
                plt.imshow(images[i], vmin=0, vmax=1)  # Display normalized image
            else:
                plt.imshow(tf.cast(images[i] * 255, tf.uint8))  # Display non-normalized image
            plt.title(class_names[labels[i].numpy()])
            plt.axis("off")
    plt.savefig(filename, facecolor=fig.get_facecolor(), edgecolor='none')

def save_example_images():
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
            plt.savefig("example_cards.jpg")


def save_accuracy(model_history, model_epochs):
    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']

    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    epochs_range = range(model_epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig("model_accuracy.jpg")
