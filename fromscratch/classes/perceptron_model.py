from fromscratch.classes.perceptron import Perceptron


class PerceptronModel():

    def __init__(self, train_x, train_y, l_rate, n_epoch) -> None:
        self.train_x = train_x
        self.train_y = train_y
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    # Estimate Perceptron weights using stochastic gradient descent
    def train(self):
        n_inputs = len(self.train_x[0])
        perceptron = Perceptron(n_inputs)

        for epoch in range(self.n_epoch):
            sum_error = 0.0

            for X_idx, X in enumerate(self.train_x):
                prediction = perceptron.predict(X)
                error = self.train_y[X_idx] - prediction

                perceptron.update_model_weights(X, self.l_rate, error)

                sum_error += error ** 2

            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, self.l_rate, sum_error))

        return perceptron.weights, perceptron.bias
