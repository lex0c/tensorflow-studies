# Digits handwritten classification

TensorFlow is used to build and train a CNN on the MNIST dataset to classify handwritten digits.

## Architecture

- Two convolutional layers, each followed by batch normalization and max pooling.
- A dropout layer to prevent overfitting.
- Dense layers at the end for classification.

Regularization is applied in convolutional and dense layers to improve generalization.

## Callbacks

- `EarlyStopping` to stop training when the validation accuracy stops improving.
- `ModelCheckpoint` to save the best model based on validation accuracy.
- `ReduceLROnPlateau` to reduce the learning rate when the validation loss plateaus.
- `TensorBoard` for visualizing training progress and performance.

## Dataset

The dataset used is the MNIST dataset, which consists of 60,000 training images and 10,000 testing images of handwritten digits, each image being 28x28 pixels.

## Run

Install deps:
```sh
pip install -r requirements.txt
```

Train the model:
```sh
python train.py
```

Run TensorBoard:
```sh
tensorboard --logdir=logs/fit
```

Predict:
```sh
python run.py
```

