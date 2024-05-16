from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from predict_server.datasets import *
from tensorflow.keras import regularizers
from tensorflow.keras.applications import MobileNetV2


def augment_data():
    return keras.Sequential(
        [
            # layers.RandomRotation(0.03),
            # layers.RandomZoom(0.02),
            # layers.RandomContrast(0.2),
            # layers.RandomTranslation(0.05, 0.05),
            layers.GaussianNoise(0.005)
        ]
    )


def define_model():
    # Load the pre-trained model
    base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')

    # Freeze the base model
    base_model.trainable = False

    num_classes = len(class_names)
    model = Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(2048, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(num_classes, activation='softmax', name="outputs")
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model

