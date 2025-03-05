import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

inputs = np.array([0.05, 0.10])

weights = {
    'w1': 0.15, 'w2': 0.20, 'w3': 0.25, 'w4': 0.30,
    'w5': 0.40, 'w6': 0.45, 'w7': 0.50, 'w8': 0.55
}
b1 = 0.35
b2 = 0.60

targets = np.array([0.01, 0.99])

h1_input = inputs[0] * weights['w1'] + inputs[1] * weights['w2'] + b1
h2_input = inputs[0] * weights['w3'] + inputs[1] * weights['w4'] + b1

h1_output = sigmoid(h1_input)
h2_output = sigmoid(h2_input)

o1_input = h1_output * weights['w5'] + h2_output * weights['w6'] + b2
o2_input = h1_output * weights['w7'] + h2_output * weights['w8'] + b2

o1_output = sigmoid(o1_input)
o2_output = sigmoid(o2_input)

error_o1 = 0.5 * (targets[0] - o1_output) ** 2
error_o2 = 0.5 * (targets[1] - o2_output) ** 2
total_error = error_o1 + error_o2

delta_o1 = (o1_output - targets[0]) * sigmoid_derivative(o1_output)
delta_o2 = (o2_output - targets[1]) * sigmoid_derivative(o2_output)

delta_h1 = (delta_o1 * weights['w5'] + delta_o2 * weights['w7']) * sigmoid_derivative(h1_output)
delta_h2 = (delta_o1 * weights['w6'] + delta_o2 * weights['w8']) * sigmoid_derivative(h2_output)

learning_rate = 0.5
weights['w5'] -= learning_rate * delta_o1 * h1_output
weights['w6'] -= learning_rate * delta_o1 * h2_output
weights['w7'] -= learning_rate * delta_o2 * h1_output
weights['w8'] -= learning_rate * delta_o2 * h2_output

weights['w1'] -= learning_rate * delta_h1 * inputs[0]
weights['w2'] -= learning_rate * delta_h1 * inputs[1]
weights['w3'] -= learning_rate * delta_h2 * inputs[0]
weights['w4'] -= learning_rate * delta_h2 * inputs[1]

b2 -= learning_rate * (delta_o1 + delta_o2)
b1 -= learning_rate * (delta_h1 + delta_h2)

print(f"Hidden Layer Outputs: h1={h1_output:.4f}, h2={h2_output:.4f}")
print(f"Output Layer Outputs: o1={o1_output:.4f}, o2={o2_output:.4f}")
print(f"Total Error: {total_error:.6f}")