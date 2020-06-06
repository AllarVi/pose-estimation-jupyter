from unittest import TestCase

from fromscratch.classes.perceptron import Perceptron


class TestPerceptron(TestCase):

    def setUp(self):
        self.perceptron = Perceptron()

    def test_predict(self):
        activation = self.perceptron.predict([2.7810836, 2.550537003], [0.0, 0.0], 0.0)

        self.assertEqual(1.0, activation)
