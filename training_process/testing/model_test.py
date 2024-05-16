from tensorflow.keras.models import load_model
from datasets import test_ds
import tensorflow as tf


def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


# Load the model
model = load_model('experiments/model_8/epochs-97-small-augm-93valacc/kog.keras')

# Assuming that `test_ds` is your test dataset
test_ds = test_ds.map(normalize)
test_loss, test_accuracy = model.evaluate(test_ds)

print(f"Test accuracy: {test_accuracy}")
print(f"Test loss: {test_loss}")

# without normalising
# 17/17 ━━━━━━━━━━━━━━━━━━━━ 2s 52ms/step - accuracy: 0.6623 - loss: 2.0999
# Test accuracy: 0.645283043384552
# Test loss: 2.16107439994812

# after normalising
# 17/17 ━━━━━━━━━━━━━━━━━━━━ 2s 52ms/step - accuracy: 0.9284 - loss: 0.3941
# Test accuracy: 0.9320755004882812
# Test loss: 0.41133245825767517
