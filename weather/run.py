import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


model = load_model('weather.keras')


def generate_weather_dataset(num_samples=100):
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


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    #plt.show()
    plt.savefig('confusion-matrix.png', dpi=300)


test_features, test_labels, test_labels_raw = generate_weather_dataset(100)
test_labels_normalized = tf.keras.utils.to_categorical(test_labels, num_classes=len(test_labels_raw))


new_weather_data = [8, 55, 12]  # Example: [Temperature, Humidity, Wind Speed]
predicted_condition = model.predict(np.array([new_weather_data]))
print("Weather input (temperature, humidity, wind speed):", new_weather_data)
print("Predicted weather condition:", test_labels_raw[np.argmax(predicted_condition)])


predictions = model.predict(test_features)
predicted_labels = np.argmax(predictions, axis=1)

#true_labels = np.argmax(test_labels_normalized, axis=1)  # Converting one-hot encoded labels back to class indices

cm = confusion_matrix(test_labels, predicted_labels)

class_names = test_labels_raw
plot_confusion_matrix(cm, class_names)

