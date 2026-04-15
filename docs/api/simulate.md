# Training and Simulation for Brittle Star Models

## Simulating a model

In order to simulate and view the behavior of a trained model, you can use the `simulate.py` script. This script allows you to specify the path to a trained model and will launch a simulation using that model. This script has the following parameters:

- `--model`: The path to the trained model artifact to simulate.
- `--model-type`: The type of model to simulate (e.g., `random`, ...)
- `--task`: The task to simulate (e.g., `directed_locomotion`, ...)
- `--seed`: The random seed for reproducibility.

```bash
python simulate.py --model artifacts/my_model --model-type random --task directed_locomotion --seed 0
```