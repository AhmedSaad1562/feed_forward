import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

inputs = np.array([0.05, 0.10])

weights = {
    'w1': 0.15, 'w2': 0.20, 'w3': 0.25, 'w4': 0.30,
    'w5': 0.40, 'w6': 0.45, 'w7': 0.50, 'w8': 0.55
}
b1 = 0.35
b2 = 0.60

h1_input = inputs[0] * weights['w1'] + inputs[1] * weights['w2'] + b1
h2_input = inputs[0] * weights['w3'] + inputs[1] * weights['w4'] + b1

h1_output = sigmoid(h1_input)
h2_output = sigmoid(h2_input)

o1_input = h1_output * weights['w5'] + h2_output * weights['w6'] + b2
o2_input = h1_output * weights['w7'] + h2_output * weights['w8'] + b2

o1_output = sigmoid(o1_input)
o2_output = sigmoid(o2_input)

print(f"Hidden Layer Outputs: h1={h1_output:.4f}, h2={h2_output:.4f}")
print(f"Output Layer Outputs: o1={o1_output:.4f}, o2={o2_output:.4f}")
