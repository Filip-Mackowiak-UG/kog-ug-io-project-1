from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from predict_server.datasets import *
from tensorflow.keras import regularizers


def augment_data():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.03),
            layers.RandomZoom(0.02),
            layers.RandomContrast(0.2),
            layers.RandomTranslation(0.1, 0.1),
        ]
    )


def define_model():
    num_classes = len(class_names)
    model = Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        augment_data(),
        layers.Rescaling(1. / 255),
        layers.Resizing(IMG_HEIGHT, IMG_WIDTH),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dense(num_classes, name="outputs")
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model
