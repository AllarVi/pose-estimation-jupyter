class FeedForwardNetworkModel:

    def __init__(self, train, l_rate, n_epoch) -> None:
        self.train = train
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    def train_network(self, network, n_outputs):
        for epoch in range(self.n_epoch):
            sum_error = 0

            for row in self.train:
                outputs = network.forward_propagate(row)

                expected = [0 for i in range(n_outputs)]
                expected[row[-1]] = 1

                sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])

                network.backward_propagate_error(expected)
                network.update_network_weights(row, self.l_rate)

            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, self.l_rate, sum_error))
