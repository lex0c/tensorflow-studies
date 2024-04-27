import tensorflow as tf
import numpy as np
from tensorflow import keras


xs = np.random.uniform(low=-10.0, high=10.0, size=100)
ys = np.array([5*x+1 for x in xs]) # Y = 5X + 1
#print(xs, ys)


model = tf.keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(xs, ys, epochs=500)

model.save('simple_linear_function.h5')

