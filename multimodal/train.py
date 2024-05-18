import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Flatten, Conv2D, MaxPooling2D, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import fashion_mnist, imdb
from tensorflow.keras.utils import to_categorical
import numpy as np


# Load Fashion MNIST dataset
(fashion_train_images, fashion_train_labels), (fashion_test_images, fashion_test_labels) = fashion_mnist.load_data()

# Preprocess images
fashion_train_images = fashion_train_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
fashion_test_images = fashion_test_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

# Convert labels to one-hot encoding
fashion_train_labels = to_categorical(fashion_train_labels, 10)
fashion_test_labels = to_categorical(fashion_test_labels, 10)

# Load IMDb dataset
num_words = 10000
maxlen = 100

(imdb_train_data, imdb_train_labels), (imdb_test_data, imdb_test_labels) = imdb.load_data(num_words=num_words)

# Preprocess text data
imdb_train_data = pad_sequences(imdb_train_data, maxlen=maxlen)
imdb_test_data = pad_sequences(imdb_test_data, maxlen=maxlen)

# Convert labels to one-hot encoding
imdb_train_labels = to_categorical(imdb_train_labels, 2)
imdb_test_labels = to_categorical(imdb_test_labels, 2)

# Create smaller matched dataset
num_samples = min(len(fashion_train_images), len(imdb_train_data))
fashion_train_images = fashion_train_images[:num_samples]
fashion_train_labels = fashion_train_labels[:num_samples]
imdb_train_data = imdb_train_data[:num_samples]
imdb_train_labels = imdb_train_labels[:num_samples]

num_test_samples = min(len(fashion_test_images), len(imdb_test_data))
fashion_test_images = fashion_test_images[:num_test_samples]
fashion_test_labels = fashion_test_labels[:num_test_samples]
imdb_test_data = imdb_test_data[:num_test_samples]
imdb_test_labels = imdb_test_labels[:num_test_samples]

# Image input model
image_input = Input(shape=(28, 28, 1), name='image_input')
conv1 = Conv2D(64, (3, 3), activation='relu')(image_input)
pool1 = MaxPooling2D((2, 2))(conv1)
conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
pool2 = MaxPooling2D((2, 2))(conv2)
flatten = Flatten()(pool2)
image_dense = Dense(64, activation='relu')(flatten)

# Text input model
text_input = Input(shape=(100,), name='text_input')
embedding = Embedding(input_dim=num_words, output_dim=64)(text_input)
lstm1 = LSTM(64, return_sequences=True)(embedding)
lstm2 = LSTM(64)(lstm1)
text_dense = Dense(64, activation='relu')(lstm2)

# Shared Dense layer
shared_dense = Dense(64, activation='relu')

# Apply the shared dense layer
image_shared_output = shared_dense(image_dense)
text_shared_output = shared_dense(text_dense)

# Combined model
combined = Concatenate()([image_shared_output, text_shared_output])
combined_dense1 = Dense(64, activation='relu')(combined)
dropout1 = Dropout(0.2)(combined_dense1)
combined_dense2 = Dense(64, activation='relu')(dropout1)
combined_output = Dense(12, activation='softmax')(combined_dense2)

model = Model(inputs=[image_input, text_input], outputs=combined_output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


combined_train_labels = np.concatenate((fashion_train_labels, imdb_train_labels), axis=1)
combined_test_labels = np.concatenate((fashion_test_labels, imdb_test_labels), axis=1)

model.fit(
    [fashion_train_images, imdb_train_data],
    combined_train_labels,
    epochs=10,
    batch_size=32,
    validation_data=([fashion_test_images, imdb_test_data], combined_test_labels)
)


loss, accuracy = model.evaluate([fashion_test_images, imdb_test_data], combined_test_labels)
print(f"Loss: {loss}, Accuracy: {accuracy}")

predictions = model.predict([fashion_test_images, imdb_test_data])


# Print first 5 predictions
print("First 5 predictions:")
print(predictions[:5])

# Print corresponding true labels
print("Corresponding true labels:")
print(combined_test_labels[:5])


