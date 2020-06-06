from random import seed

from fromscratch.classes.feed_forward_network import FeedForwardNetwork


class Main:

    @staticmethod
    def run():
        seed(1)
        network = FeedForwardNetwork(2, 1, 2)
        network.summary()

        row = [1, 0, None]
        output = network.forward_propagate(row)
        print(output)


if __name__ == '__main__':
    Main.run()
