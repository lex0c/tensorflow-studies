import tensorflow as tf
import numpy as np


def generate_weather_dataset(num_samples=1000):
    weather_data = {
        'Sunny': {'temp': (20, 35), 'humidity': (10, 50), 'wind': (0, 20)},
        'Cloudy': {'temp': (10, 20), 'humidity': (50, 90), 'wind': (0, 10)},
        'Rainy': {'temp': (5, 15), 'humidity': (80, 100), 'wind': (10, 25)},
        'Snowy': {'temp': (-5, 0), 'humidity': (50, 100), 'wind': (5, 15)}
    }

    features = []
    labels = []

    # Define label indices
    label_indices = {label: idx for idx, label in enumerate(weather_data.keys())}

    # Generate samples
    for label, ranges in weather_data.items():
        for _ in range(num_samples):
            temp = np.random.uniform(*ranges['temp'])
            humidity = np.random.uniform(*ranges['humidity'])
            wind = np.random.uniform(*ranges['wind'])

            features.append([temp, humidity, wind])
            labels.append(label_indices[label])

    # Convert to numpy arrays
    features = np.array(features)

    return features, labels, list(weather_data.keys())


features, labels, labels_raw = generate_weather_dataset(500)
labels_normalized = tf.keras.utils.to_categorical(labels, num_classes=len(labels_raw))

test_features, test_labels, _ = generate_weather_dataset(50)
test_labels_normalized = tf.keras.utils.to_categorical(test_labels, num_classes=len(labels_raw))


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, input_shape=(features.shape[1],), activation='relu'),
    tf.keras.layers.Dense(len(labels_raw), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(features, labels_normalized, epochs=100, batch_size=32)

test_loss, test_accuracy = model.evaluate(test_features, test_labels_normalized)

print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss*100:.2f}%")

model.save('weather.keras')

