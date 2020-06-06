class FeedForwardNetworkModel:

    def __init__(self, train_x, train_y, l_rate, n_epoch) -> None:
        self.train_x = train_x
        self.train_y = train_y
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    def train_network(self, network, n_outputs):
        for epoch in range(self.n_epoch):
            sum_error = 0

            for i, inputs in enumerate(self.train_x):
                outputs = network.forward_propagate(inputs)

                expected = [0 for i in range(n_outputs)]
                expected[self.train_y[i]] = 1

                sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])

                network.backward_propagate_error(expected)
                network.update_network_weights(inputs, self.l_rate)

            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, self.l_rate, sum_error))
