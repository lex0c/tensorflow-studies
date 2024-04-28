import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


model = load_model('fashion_mnist_classification.h5')

mnist = tf.keras.datasets.fashion_mnist

_, (test_images, test_labels) = mnist.load_data()

test_images = test_images / 255.0 # Normalizing values between 0 and 1

classifications = model.predict(test_images)

hits = 0

for c, label in zip(classifications, test_labels):
    if np.argmax(c, axis=0) == label:
        hits += 1


print('Total hits (%):', (hits/len(test_labels)*100))

