from random import random

from fromscratch.classes.sigmoid_unit import SigmoidUnit


class FeedForwardNetwork():

    def __init__(self, n_inputs, n_hidden, n_outputs) -> None:
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        layers = list()

        hidden_layer = self.get_dense_layer(n_inputs, n_hidden)
        layers.append(hidden_layer)

        output_layer = self.get_dense_layer(n_hidden, n_outputs)
        layers.append(output_layer)

        self.layers = layers

    def get_dense_layer(self, n_inputs, n_outputs):
        return [{'weights': [random() for i in range(n_inputs)],
                 'bias': random()}
                for i in range(n_outputs)]

    def summary(self):
        for layer in self.layers:
            print(layer)

    def forward_propagate(self, row):
        inputs = row

        sigmoid_unit = SigmoidUnit()

        for layer in self.layers:
            new_inputs = []
            for neuron in layer:
                neuron['output'] = sigmoid_unit.predict(inputs[:-1],
                                                        neuron['weights'],
                                                        neuron['bias'])
                new_inputs.append(neuron['output'])
            inputs = new_inputs

        return inputs
