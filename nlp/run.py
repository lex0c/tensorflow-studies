import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random


model = load_model('nlp.keras')

max_features = 10000
maxlen = 500

_, (x_test, y_test) = imdb.load_data(num_words=max_features)
randindex = random.randint(0, len(x_test))

# standardize string length
padded_sequence = pad_sequences([x_test[randindex]], maxlen=maxlen)

prediction = model.predict(padded_sequence)

print('Predicted:', prediction)
print('Class:', y_test[randindex])

