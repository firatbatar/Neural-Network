import numpy as np
import scipy.special


class NeuralNetwork:
    # Set the number of input, hidden and output nodes
    def __init__(self, input_nodes, hidden_nodes, output_nodes, lr):
        # Number of the nodes in each layer
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        """
        weights are initialised randomly sampling from a range that is roughly the inverse of the square root
        of the number of links into a node.
        """
        self.weights_ih = np.random.normal(0.0, pow(self.input_nodes, -0.5), (self.hidden_nodes, self.input_nodes))  # Weights from input nodes to hiddens
        self.weights_ho = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.output_nodes, self.hidden_nodes))  # Weights from hidden nodes to outputs

        self.learning_rate = lr
        self.activation_func = lambda x: scipy.special.expit(x)

    # Refine the weights after being given a training set example to learn
    def train(self, inputs_list, targets_list):
        """
        ● The first part is working out the output for a given training example.
        That is no different to what we just did with the query() function.

        ● The second part is taking this calculated output, comparing it with the
        desired output, and using the difference to guide the updating of the
        network weights.
        """

        # Same as the query

        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        # Convert targets list to 2d array
        targets = np.array(targets_list, ndmin=2).T

        # Signals into hidden nodes
        hidden_inputs = np.dot(self.weights_ih, inputs)
        # Signals from hidden nodes
        hidden_outputs = self.activation_func(hidden_inputs)

        # Signals into output nodes
        output_inputs = np.dot(self.weights_ho, hidden_outputs)
        # Signals from output nodes
        output_outputs = self.activation_func(output_inputs)


        # Error is (target - guess)
        output_errors = targets - output_outputs

        # E_h = Wt . E_o
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.weights_ho.T, output_errors)

        # Calculate the changes of weights
        # dW = lr * E * sigmoid(O) * (1 - sigmoid(O)) . Ot
        delta_ho = self.learning_rate * np.dot(output_errors * output_outputs * (1 - output_outputs), np.transpose(hidden_outputs))
        delta_ih = self.learning_rate * np.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), np.transpose(inputs))

        # Update the weights
        self.weights_ih += delta_ih
        self.weights_ho += delta_ho

    # Give an answer from the output nodes after being given an input
    def query(self, inputs_list):
        # X = W . I

        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # Signals into hidden nodes
        hidden_inputs = np.dot(self.weights_ih, inputs)
        # Signals from hidden nodes
        hidden_outputs = self.activation_func(hidden_inputs)

        # Signals into output nodes
        output_inputs = np.dot(self.weights_ho, hidden_outputs)
        # Signals from output nodes
        output_outputs = self.activation_func(output_inputs)

        return output_outputs
