import numpy as np


class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size):
        self.weights_hidden = np.random.rand(hidden_size, input_size)
        self.hidden_bias = np.random.rand(hidden_size)
        self.weights_output = np.random.rand(output_size, hidden_size)
        self.output_bias = np.random.rand(output_size)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def d_sigmoid(self, z):
        return z * (1 - z)

    #def d_J(self, a, y):
     #   return (a - y) / np.dot(a, (1 + a))

    def forward(self, inputs):
        hidden_outputs = self.sigmoid(np.dot(self.weights_hidden, inputs) + self.hidden_bias)
        output = self.sigmoid(np.dot(self.weights_output, hidden_outputs) + self.output_bias)
        return hidden_outputs, output

    def backward(self, inputs, y, lr):
        hidden_outputs, output = self.forward(inputs)
        #d_a2 = self.d_J(output, y)
        d_z2 = output - y
        #d_z2 = d_a2 * self.d_sigmoid(output)
        d_w2 = np.outer(d_z2, hidden_outputs.T)
        d_b2 = d_z2
        self.weights_output = self.weights_output - (lr * d_w2)
        self.output_bias = self.output_bias - (lr * d_b2)

        d_a1 = np.dot(self.weights_output.T, d_z2)
        d_z1 = d_a1 * self.d_sigmoid(hidden_outputs)
        d_w1 = np.outer(d_z1, inputs.T)
        d_b1 = d_z1

        self.weights_hidden = self.weights_hidden - (lr * d_w1)
        self.hidden_bias = self.hidden_bias - (lr * d_b1)


nn = NeuralNetwork(2, 3, 2)

inputs = np.array([[0, 1]])
y = np.array([[0, 1]])

learning_rate = 0.25

for _ in range(5000):
    for i in range(len(inputs)):
       nn.backward(inputs[i], y[i], learning_rate)

for i in range(len(inputs)):
    hidden_outputs, output = nn.forward(inputs[i])
    print(f"Entrada: {inputs[i]} - Salida: {output}\n")

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0, 0], [1, 0], [1, 0], [0, 1]])

for epoch in range(5000):
    for i in range(len(inputs)):
        nn.backward(inputs[i], y[i], learning_rate)

for i in range(len(inputs)):
    hidden_outputs, output = nn.forward(inputs[i])
    print(f"Entrada: {inputs[i]} - Salida: {output}")
