# importing required libraries
import numpy as np

# defining sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))


# defining a neural network class
class neuralNetwork:

    # initializing the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate=0.1):
        self.input_nodes = inputnodes
        self.hidden_nodes = hiddennodes
        self.output_nodes = outputnodes

        # learning rate
        self.learning_rate = learningrate

        # initializing the neural network with random weights to go from input layer to
        # hidden layer and from hidden layer to output layer

        self.weight_input_hidden = np.random.randn(self.hidden_nodes, self.input_nodes) * 0.01
        self.weight_hidden_output = np.random.randn(self.output_nodes, self.hidden_nodes) * 0.01

        # calculating the activation function (sigmoid)
        self.sigmoid = lambda x: sigmoid(x)
        pass

    def train(self, inputs_list, target_list):
        # flatten
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        # input times the weights
        hidden_inputs = np.dot(self.weight_input_hidden, inputs)

        # apply the sigmoid function
        hidden_outputs = self.sigmoid(hidden_inputs)

        # same thing as above but for the output
        final_inputs = np.dot(self.weight_hidden_output, hidden_outputs)
        final_outputs = self.sigmoid(final_inputs)

        # calculating the error
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.weight_hidden_output.T, output_errors)

        # BACKPROPOGATION
        self.weight_hidden_output += self.learning_rate * np.dot((output_errors * final_outputs
                                                                  * (1.0 - final_outputs)),
                                                                 np.transpose(hidden_outputs))
        self.weight_input_hidden += self.learning_rate * np.dot((hidden_errors * hidden_outputs
                                                                 * (1.0 - hidden_outputs)),
                                                                np.transpose(inputs))

        pass

    def predict(self, inputs_list):
        # flatten
        inputs = np.array(inputs_list, ndmin=2).T
        # input times the weights
        hidden_inputs = np.dot(self.weight_input_hidden, inputs)
        # apply the sigmoid function
        hidden_outputs = self.sigmoid(hidden_inputs)

        # same thing as above but for the output
        final_inputs = np.dot(self.weight_hidden_output, hidden_outputs)
        final_outputs = self.sigmoid(final_inputs)

        return final_outputs
