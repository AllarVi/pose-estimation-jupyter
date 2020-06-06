from random import seed
from unittest import TestCase

from fromscratch.classes.feed_forward_network import FeedForwardNetwork


class TestFeedForwardNetwork(TestCase):

    def setUp(self):
        seed(1)
        self.network = FeedForwardNetwork(2, 1, 2)

    def test_forward_propagate(self):
        output = self.network.forward_propagate([1, 0, None])

        expected_output = [0.6213859615555266, 0.6573693455986976]

        for i, o in enumerate(output):
            self.assertEqual(expected_output[i], o)
