import random

import math

# Training Data
x = [[0,0],[1,0],[0,1],[1,1]]

# Expected Outputs
y = [0,1,1,0]

# Activation Function and its Derivative
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize Weights and Biases
W1 = [[random.uniform(0, 1) for i in range(2)] for i in range(2)] 
b1 = [random.uniform(0, 1) for i in range(2)]
W2 = [random.uniform(0, 1) for i in range(2)]
b2 = random.uniform(0, 1)
learning_rate = 0.5

# Training Loop
for epoch in range(10000):
    total_loss = 0

    for i in range(len(x)):
        #Training for each data point
        inputs = x[i]
        target = y[i]
    
        hidden_input = [0,0]
        hidden_output = [0,0]

        # Do this for each neuron in hidden layer
        for j in range(2):
            #Multiply inputs by weights and add bias
            hidden_input[j] = inputs[0]*W1[0][j] + inputs[1]*W1[1][j] + b1[j]

            #Apply activation function
            hidden_output[j] = sigmoid(hidden_input[j])

        # Calculate final output
        final_input = hidden_output[0]*W2[0] + hidden_output[1]*W2[1] + b2
        final_output = sigmoid(final_input)

        # Calculate error
        error = target - final_output
        total_loss += error**2

        # Backpropagation
        d_output = error * sigmoid_derivative(final_output)

        d_hidden = [0,0]

        for j in range(2):
            d_hidden[j] = d_output * W2[j] * sigmoid_derivative(hidden_output[j])

        for j in range(2):
            W2[j] += learning_rate * d_output * hidden_output[j]

        b2 += learning_rate * d_output

        #Update weights and biases for hidden layer
        for j in range(2):
            for k in range(2):
                W1[k][j] += learning_rate * d_hidden[j] * inputs[k]
            b1[j] += learning_rate * d_hidden[j]

        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Data: {inputs}, Output: {final_output} Loss: {total_loss}')

# Prediction after training
def predict(inputs):
    hidden_input = [0,0]
    hidden_output = [0,0]

    for j in range(2):
        hidden_input[j] = inputs[0]*W1[0][j] + inputs[1]*W1[1][j] + b1[j]
        hidden_output[j] = sigmoid(hidden_input[j])

    final_input = hidden_output[0]*W2[0] + hidden_output[1]*W2[1] + b2
    final_output = sigmoid(final_input)

    return final_output

tests = [[0,0],[1,0],[0,1],[1,1]]

for t in tests:
    print(t, ":", predict(t))

