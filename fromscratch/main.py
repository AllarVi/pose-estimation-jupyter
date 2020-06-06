from fromscratch.classes.perceptron import Perceptron


class Main:

    @staticmethod
    def run():
        # Calculate weights
        dataset = [[2.7810836, 2.550537003, 0],
                   [1.465489372, 2.362125076, 0],
                   [3.396561688, 4.400293529, 0],
                   [1.38807019, 1.850220317, 0],
                   [3.06407232, 3.005305973, 0],
                   [7.627531214, 2.759262235, 1],
                   [5.332441248, 2.088626775, 1],
                   [6.922596716, 1.77106367, 1],
                   [8.675418651, -0.242068655, 1],
                   [7.673756466, 3.508563011, 1]]
        l_rate = 0.1
        n_epoch = 5

        train_x = [row[0:-1] for row in dataset]
        train_y = [row[-1] for row in dataset]

        weights, bias = Main.train_weights(train_x, train_y, l_rate, n_epoch)

        print(weights, bias)

    # Estimate Perceptron weights using stochastic gradient descent
    @staticmethod
    def train_weights(train_x, train_y, l_rate, n_epoch):

        weights = [0.0 for i in range(len(train_x[0]))]
        bias = 0.0

        perceptron = Perceptron()

        for epoch in range(n_epoch):
            sum_error = 0.0

            for X_idx, X in enumerate(train_x):
                prediction = perceptron.predict(X, weights, bias)
                error = train_y[X_idx] - prediction

                bias = Main.update_bias(error, l_rate, bias)
                weights = Main.update_weights(error, l_rate, weights, X)

                sum_error += error ** 2

            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

        return weights, bias

    @staticmethod
    def update_weights(error, l_rate, weights, X):
        return [weights[i] + l_rate * error * X[i] for i in range(len(X))]

    @staticmethod
    def update_bias(error, l_rate, bias):
        return bias + l_rate * error


if __name__ == '__main__':
    Main.run()
