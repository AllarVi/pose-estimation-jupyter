from random import seed

from fromscratch.classes.feed_forward_network import FeedForwardNetwork
from fromscratch.classes.feed_forward_network_model import FeedForwardNetworkModel


class Main:

    @staticmethod
    def run():
        seed(1)
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

        train_x = [row[0:-1] for row in dataset]
        train_y = [row[-1] for row in dataset]

        n_inputs = len(train_x[0])
        n_outputs = len(set(train_y))

        network = FeedForwardNetwork(n_inputs, 2, n_outputs)

        network_model = FeedForwardNetworkModel(train_x, train_y, l_rate=0.5, n_epoch=20)
        network_model.train_network(network, n_outputs)

        network.summary()


if __name__ == '__main__':
    Main.run()
