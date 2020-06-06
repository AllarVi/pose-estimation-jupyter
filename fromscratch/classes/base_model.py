class BaseModel:
    @staticmethod
    def update_weights(error, l_rate, weights, X):
        return [weights[i] + l_rate * error * X[i] for i in range(len(X))]

    @staticmethod
    def update_bias(error, l_rate, bias):
        return bias + l_rate * error
