import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


model = load_model('img_number_classification.keras')

mnist = tf.keras.datasets.mnist

_, (test_images, test_labels) = mnist.load_data()

test_images = test_images / 255.0 # Normalizing values between 0 and 1
test_images = np.expand_dims(test_images, axis=-1) # Normalization of dimensions to 4D

classifications = model.predict(test_images)

hits = 0

for c, label in zip(classifications, test_labels):
    if np.argmax(c) == label:
        hits += 1


print('Total hits (%):', (hits/len(test_labels)*100))

