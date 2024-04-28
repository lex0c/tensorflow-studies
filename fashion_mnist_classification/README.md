# Fashion mnist classification

TensorFlow is used to build and train a machine learning model to predict img class.

## Architecture

- Sequential model.
- Flatten layer to convert img to one dimensional vector.
- Two dense layers with 256 neurons.
- Adam optimizer.
- Loss function is sparse categorical crossentropy.

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

