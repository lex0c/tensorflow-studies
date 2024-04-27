import datetime
import tensorflow as tf
import numpy as np
from tensorflow import keras


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

xs = np.random.uniform(low=-10.0, high=10.0, size=100)
ys = np.array([5*x+1 for x in xs]) # Y = 5X + 1
#print(xs, ys)


model = tf.keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

model.fit(xs, ys, epochs=500, callbacks=[tensorboard_callback])

model.save('simple_linear_function.h5')

