from random import seed

from fromscratch.classes.feed_forward_network import FeedForwardNetwork


class Main:

    @staticmethod
    def run():
        seed(1)
        network = FeedForwardNetwork(2, 1, 2)
        network.summary()

        row = [1, 0, None]
        output = network.forward_propagate(row)
        print(f"Output: {output}")

        expected = [0, 1]
        Main.backward_propagate_error(network.layers, expected)
        network.summary()

    # Calculate the derivative of an neuron output
    @staticmethod
    def transfer_derivative(output):
        return output * (1.0 - output)

    # Backpropagate error and store in neurons
    @staticmethod
    def backward_propagate_error(layers, expected):
        for i in reversed(range(len(layers))):
            layer = layers[i]
            errors = list()

            last_layer_idx = len(layers) - 1
            if Main.is_last_layer(i, last_layer_idx):
                errors = Main.get_last_layer_errors(expected, layer)
            else:
                for current_layer_neuron_idx in range(len(layer)):
                    prev_layer = layers[i + 1]
                    error = Main.get_error_for_current_layer_neuron(current_layer_neuron_idx,
                                                                    prev_layer)
                    errors.append(error)

            for j, neuron in enumerate(layer):
                neuron['delta'] = errors[j] * Main.transfer_derivative(neuron['output'])

    @staticmethod
    def get_error_for_current_layer_neuron(current_layer_neuron_idx, prev_layer):
        return sum([prev_layer_neuron['weights'][current_layer_neuron_idx] * prev_layer_neuron['delta']
                    for prev_layer_neuron in prev_layer])

    @staticmethod
    def get_last_layer_errors(expected, layer):
        return [(expected[i] - neuron['output']) for i, neuron in enumerate(layer)]

    @staticmethod
    def is_last_layer(current_layer_idx, last_layer_idx):
        return current_layer_idx == last_layer_idx


if __name__ == '__main__':
    Main.run()
