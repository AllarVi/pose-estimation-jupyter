class Perceptron:

    def predict(self, x, weights, bias):
        state = self.get_state(bias, x, weights)

        return self.get_activation(state)

    def get_activation(self, state):
        return 1.0 if state >= 0.0 else 0.0

    def get_state(self, bias, x, weights):
        weighted_input = self.get_weighted_input(x, weights)

        state = weighted_input + bias

        return state

    def get_weighted_input(self, x, weights):
        return sum([weights[i] * x[i] for i in range(len(x))])
