from math import exp

from fromscratch.classes.perceptron import Perceptron


class SigmoidUnit(Perceptron):

    def get_activation(self, state):
        return 1.0 / (1.0 + exp(-state))
