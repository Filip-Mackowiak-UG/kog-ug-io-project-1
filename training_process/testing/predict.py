from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from constants import *
from datasets import class_names
import numpy as np
from PIL import Image


# Load the model
model = load_model('experiments/model_8/epochs-97-small-augm-93valacc/kog.keras')

# Load the image
img = load_img('my_data/white_bg/4_of_clubs_cropped.jpg', target_size=(IMG_HEIGHT, IMG_WIDTH))

# Save the image
img.save('preprocessed_image.jpg')

# Convert the image to a numpy array
img_array = img_to_array(img)

# Normalize the image array
img_array = img_array / 255.0

# Add an extra dimension for the batch size
img_array = tf.expand_dims(img_array, 0)

# Make the prediction
prediction = model.predict(img_array)

# Get the predicted class
predicted_class = class_names[np.argmax(prediction)]

print(f"The model predicts that the image is a: {predicted_class}")
