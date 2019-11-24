import pandas as pd
from back_propagation import generate_random_weights, propagate_forward, propagate_backward

# Constants
INPUT_NUM = 784
HIDDEN_NUM = 16
OUTPUT_NUM = 10
LEARNING_RATE = 0.005
EPOCH = 75
TRAINING_INSTANCES = 5000
TESTING_INSTANCES = 10000

input_df = pd.read_csv("data/training60000.csv", header=None)
output_df = pd.read_csv("data/training60000_labels.csv", header=None)
weights = generate_random_weights(INPUT_NUM, HIDDEN_NUM, OUTPUT_NUM)

# Build model using back propagation algorithm
for epoch in range(EPOCH):
    network_error = 0
    for i in range(TRAINING_INSTANCES):
        # Update input and output
        inputs = list(input_df.iloc[i])
        inputs.insert(0, 1.0)
        target_outputs = [0.01] * OUTPUT_NUM
        target_outputs[next(iter(output_df.iloc[i]))] = 0.99

        # Get outputs and update weights
        predicted_outputs = propagate_forward(inputs, weights)
        network_error += propagate_backward(inputs, predicted_outputs, target_outputs, LEARNING_RATE, weights)
    print("Network Error at epoch {0}:".format(epoch + 1), network_error)

# Test model using test data set
input_df = pd.read_csv("data/testing10000.csv", header=None)
output_df = pd.read_csv("data/testing10000_labels.csv", header=None)

correct_classification = 0
for i in range(TESTING_INSTANCES):
    inputs = list(input_df.iloc[i])
    inputs.insert(0, 1.0)
    target_label = next(iter(output_df.iloc[i]))
    outputs = propagate_forward(inputs, weights)

    # If target output equals output
    if target_label == outputs["output"].index(max(outputs["output"])):
        correct_classification += 1

# Print test results
print("==== Results")
print("Network properties: Input: {0}, Hidden: {1}, Output: {2}".format(INPUT_NUM, HIDDEN_NUM, OUTPUT_NUM))
print("Learning rate: {0}, Epoch: {1}".format(LEARNING_RATE, EPOCH))
print("Correct classification = ", correct_classification)
print("Incorrect classification = ", TESTING_INSTANCES - correct_classification)
print("Accuracy = ", (correct_classification / TESTING_INSTANCES) * 100, "%")
