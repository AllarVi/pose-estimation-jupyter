from fromscratch.classes.base_model import BaseModel
from fromscratch.classes.sigmoid_unit import SigmoidUnit


class FeedForwardNetwork(BaseModel):

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
        return [SigmoidUnit(n_inputs, rand=True)
                for i in range(n_outputs)]

    def summary(self):
        for i, layer in enumerate(self.layers):
            for j, neuron in enumerate(layer):
                print(f"Layer({i}) neuron({j}) weights={neuron.weights}, bias={neuron.bias}")

    def forward_propagate(self, inputs):
        for layer in self.layers:
            new_inputs = []
            for neuron in layer:
                neuron.output = neuron.predict(inputs)

                new_inputs.append(neuron.output)
            inputs = new_inputs

        return inputs

    # Calculate the derivative of an neuron output
    @staticmethod
    def transfer_derivative(output):
        return output * (1.0 - output)

    # Backpropagate error and store in neurons (done for all neurons in network)
    def backward_propagate_error(self, expected):
        layers = self.layers

        for i in reversed(range(len(layers))):
            layer = layers[i]
            errors = list()

            last_layer_idx = len(layers) - 1
            if FeedForwardNetwork.is_last_layer(i, last_layer_idx):
                errors = FeedForwardNetwork.get_last_layer_errors(expected, layer)
            else:
                for current_layer_neuron_idx in range(len(layer)):
                    prev_layer = layers[i + 1]
                    error = FeedForwardNetwork.get_error_for_current_layer_neuron(current_layer_neuron_idx,
                                                                                  prev_layer)
                    errors.append(error)

            for j, neuron in enumerate(layer):
                neuron.delta = errors[j] * FeedForwardNetwork.transfer_derivative(neuron.output)

    # Update network weights with error
    def update_model_weights(self, inputs, l_rate):
        layers = self.layers

        for i, layer in enumerate(layers):
            # If not first layer, get neuron outputs of previous layer as inputs
            if i != 0:
                prev_layer = layers[i - 1]
                inputs = [neuron.output for neuron in prev_layer]

            for neuron in layer:
                neuron.update_neuron_weights(inputs, l_rate, neuron.delta)

    @staticmethod
    def get_error_for_current_layer_neuron(current_layer_neuron_idx, prev_layer):
        return sum([prev_layer_neuron.weights[current_layer_neuron_idx] * prev_layer_neuron.delta
                    for prev_layer_neuron in prev_layer])

    @staticmethod
    def get_last_layer_errors(expected, layer):
        return [(expected[i] - neuron.output) for i, neuron in enumerate(layer)]

    @staticmethod
    def is_last_layer(current_layer_idx, last_layer_idx):
        return current_layer_idx == last_layer_idx
