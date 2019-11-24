import pandas as pd
import numpy as np


# Returns a dictionary that contains DataFrames of randomly generated weights between -0.05 and 0.05
def generate_random_weights(input_num: int, hidden_num: int, output_num: int) -> dict:
    weights = {"input": pd.DataFrame((np.random.rand(input_num + 1, hidden_num) - 0.5) * 0.1),
               "hidden": pd.DataFrame((np.random.rand(hidden_num + 1, output_num) - 0.5) * 0.1)}
    return weights


# Returns a dictionary that contains outputs of the hidden and output layer
def propagate_forward(inputs: list, weights: pd.DataFrame) -> dict:
    outputs = {}
    hidden_outputs = list(sigmoid(inputs @ weights["input"]))
    outputs["hidden"] = hidden_outputs.copy()
    hidden_outputs.insert(0, 1.0)
    outputs["output"] = list(sigmoid(hidden_outputs @ weights["hidden"]))
    return outputs


# Updates the weights
def propagate_backward(inputs: list, outputs: list, target_outputs: list, learning_rate: float,
                       weights: pd.DataFrame) -> float:
    # Calculate error in hidden and output layer
    errors = {"output": [output * (1 - output) for output in outputs["output"]] * (
            np.array(target_outputs) - np.array(outputs["output"]))}
    errors["hidden"] = [output * (1 - output) for output in outputs["hidden"]] * \
                       np.array(weights["hidden"] @ errors["output"])[1:]

    # Update input weights
    for i in range(len(weights["input"].columns)):
        weights["input"][i] += learning_rate * errors["hidden"][i] * pd.Series(inputs)

    # Update hidden weights
    hidden_outputs = outputs["hidden"].copy()
    hidden_outputs.insert(0, 1.0)
    for i in range(len(weights["hidden"].columns)):
        weights["hidden"][i] += learning_rate * errors["output"][i] * pd.Series(hidden_outputs)

    # Return instance error
    return sum(0.5 * (np.array(target_outputs) - np.array(outputs["output"])) ** 2)


# Activation function
def sigmoid(net_value):
    return 1 / (1 + np.e ** -net_value)
