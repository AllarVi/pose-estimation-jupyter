from random import random

from fromscratch.classes.base_model import BaseModel


class Perceptron(BaseModel):

    def __init__(self, n_inputs, rand=False) -> None:
        if rand:
            self.weights = [random() for i in range(n_inputs)]
            self.bias = random()
        else:
            self.weights = [0.0 for i in range(n_inputs)]
            self.bias = 0.0

        self.output = 0.0
        self.delta = 0.0

    def predict(self, x):
        state = self.get_state(self.bias, x, self.weights)

        return self.get_activation(state)

    def get_activation(self, state):
        return 1.0 if state >= 0.0 else 0.0

    def get_state(self, bias, x, weights):
        weighted_input = self.get_weighted_input(x, weights)

        state = weighted_input + bias

        return state

    def get_weighted_input(self, x, weights):
        return sum([weights[i] * x[i] for i in range(len(x))])

    def update_model_weights(self, X, l_rate, error):
        self.weights = BaseModel.update_weights(error, l_rate, self.weights, X)
        self.bias = BaseModel.update_bias(error, l_rate, self.bias)
