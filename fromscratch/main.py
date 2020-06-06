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

        weights = Main.train_weights(dataset, l_rate, n_epoch)

        print(weights)

    # Estimate Perceptron weights using stochastic gradient descent
    @staticmethod
    def train_weights(train, l_rate, n_epoch):

        train_x = [row[0:-1] for row in train]

        weights = [0.0 for i in range(len(train_x[0]))]
        bias = 0.0

        perceptron = Perceptron()

        for epoch in range(n_epoch):
            sum_error = 0.0
            for row in train:
                prediction = perceptron.predict(row, weights, bias)
                error = row[-1] - prediction
                sum_error += error ** 2
                bias = bias + l_rate * error
                for i in range(len(row) - 1):
                    weights[i] = weights[i] + l_rate * error * row[i]
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

        return weights


if __name__ == '__main__':
    Main.run()
