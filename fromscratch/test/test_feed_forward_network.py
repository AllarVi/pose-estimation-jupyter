from random import seed
from unittest import TestCase

from fromscratch.classes.feed_forward_network import FeedForwardNetwork


class TestFeedForwardNetwork(TestCase):

    def setUp(self):
        seed(1)
        self.network = FeedForwardNetwork(2, 1, 2)

    def test_forward_propagate(self):
        output = self.network.forward_propagate([1, 0])

        expected_output = [0.6629970129852887, 0.7253160725279748]

        for i, o in enumerate(output):
            self.assertEqual(expected_output[i], o)

    def test_back_propagate_error(self):
        self.network.forward_propagate([1, 0])

        self.network.backward_propagate_error([0, 1])

        expected_deltas = [
            [-0.002711797799238243],
            [-0.14813473120687762, 0.05472601157879688]
        ]

        for i, layer in enumerate(self.network.layers):
            for j, neuron in enumerate(layer):
                self.assertEqual(expected_deltas[i][j], neuron.delta)

    def test_update_model_weights(self):
        inputs = [1, 0]
        self.network.forward_propagate(inputs)
        self.network.backward_propagate_error([0, 1])

        expected_weights_before = [
            [[0.13436424411240122, 0.8474337369372327]],
            [[0.2550690257394217], [0.4494910647887381]]
        ]
        self.assert_model_weights(expected_weights_before)

        self.network.update_model_weights(inputs, 0.5)

        expected_weights_after = [
            [[0.1330083452127821, 0.8474337369372327]],
            [[0.20243920823714898], [0.46893431066736313]]
        ]
        self.assert_model_weights(expected_weights_after)

    def assert_model_weights(self, expected_weights_after):
        for i, layer in enumerate(self.network.layers):
            for j, neuron in enumerate(layer):
                self.assertEqual(expected_weights_after[i][j], neuron.weights)
