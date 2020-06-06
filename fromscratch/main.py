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
        print(f"Output: {output}")

        expected = [0, 1]
        network.backward_propagate_error(expected)
        network.summary()


if __name__ == '__main__':
    Main.run()
