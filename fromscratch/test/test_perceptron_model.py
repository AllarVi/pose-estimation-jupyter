from unittest import TestCase

from fromscratch.classes.perceptron_model import PerceptronModel


class TestPerceptronModel(TestCase):

    def setUp(self):
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

        self.perceptron_model = PerceptronModel(train_x, train_y, l_rate, n_epoch)

    def test_predict(self):
        weights, bias = self.perceptron_model.train()

        expected_weights = [0.20653640140000007, -0.23418117710000003]
        for idx, weight in enumerate(weights):
            self.assertEqual(expected_weights[idx], weight)

        self.assertEqual(-0.1, bias)
