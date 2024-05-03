# Feelings text classification

TensorFlow is used to build and train a RNN on the IMDB dataset to classify feelings.

## Architecture

- Two LSTM layers.
- Dense layer at the end for classification.

## Callbacks

- `EarlyStopping` to stop training when the validation accuracy stops improving.
- `TensorBoard` for visualizing training progress and performance.

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

