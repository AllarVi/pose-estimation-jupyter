from fromscratch.classes.perceptron import Perceptron


class PerceptronModel:

    def __init__(self, train_x, train_y, l_rate, n_epoch) -> None:
        self.train_x = train_x
        self.train_y = train_y
        self.l_rate = l_rate
        self.n_epoch = n_epoch

        self.weights = [0.0 for i in range(len(train_x[0]))]
        self.bias = 0.0

    # Estimate Perceptron weights using stochastic gradient descent
    def train(self):
        perceptron = Perceptron()

        for epoch in range(self.n_epoch):
            sum_error = 0.0

            for X_idx, X in enumerate(self.train_x):
                prediction = perceptron.predict(X, self.weights, self.bias)
                error = self.train_y[X_idx] - prediction

                self.bias = PerceptronModel.update_bias(error, self.l_rate, self.bias)
                self.weights = PerceptronModel.update_weights(error, self.l_rate, self.weights, X)

                sum_error += error ** 2

            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, self.l_rate, sum_error))

        return self.weights, self.bias

    @staticmethod
    def update_weights(error, l_rate, weights, X):
        return [weights[i] + l_rate * error * X[i] for i in range(len(X))]

    @staticmethod
    def update_bias(error, l_rate, bias):
        return bias + l_rate * error
