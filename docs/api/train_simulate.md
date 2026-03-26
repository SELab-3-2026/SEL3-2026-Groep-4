# Training and Simulation for Brittle Star Models

## Training a model

To train a model, you can use the `train.py` script. This script allows to pass some parameters to customize the training process:

- `--out`: The output path where the trained model will be saved.
- `--model_type`: The type of model to train (e.g., `random`, ...)
- `--task`: The task to train on (e.g., `directed_locomotion`, ...)
- `--seed`: The random seed for reproducibility.
- `--epochs`: The number of epochs to train for.

This will then train the specified model on the specified task for the given number of epochs and save the trained model to the specified output path.

```bash
python train.py --out artifacts/my_model --model-type random --task directed_locomotion --seed 0 --epochs 50 
```

## Simulating a model

In order to simulate and view the behavior of a trained model, you can use the `simulate.py` script. This script allows you to specify the path to a trained model and will launch a simulation using that model. This script has the following parameters:

- `--model`: The path to the trained model artifact to simulate.
- `--model-type`: The type of model to simulate (e.g., `random`, ...)
- `--task`: The task to simulate (e.g., `directed_locomotion`, ...)
- `--seed`: The random seed for reproducibility.

```bash
python simulate.py --model artifacts/my_model --model-type random --task directed_locomotion --seed 0
```