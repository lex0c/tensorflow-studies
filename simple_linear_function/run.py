import random
import numpy as np
from tensorflow.keras.models import load_model


model = load_model('simple_linear_function.h5')

x = random.uniform(-10.0, 10.0)
expected_result = 5*x+1

print('X:', x)
print('Model predict (Y):', model.predict(np.array([x])))
print('Calculated result (Y):', expected_result)

