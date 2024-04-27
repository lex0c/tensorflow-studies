# Simple linear function prediction

TensorFlow is used to build and train a machine learning model to predict values in a simple linear function.

`Y = 5X + 1`

## Architecture

- Sequential model.
- Dense layer with one neuron and one input.
- SGD optimizer.
- Loss function is mean squared error.

## Run

Install deps:
```sh
pip install -r requirements.txt
```

Train the model:
```sh
python train.py
```
The dataset is created randomly.

Run TensorBoard:
```sh
tensorboard --logdir=logs/fit
```

Predict:
```sh
python run.py
```

