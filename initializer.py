import numpy as np

class Initializer:
    @staticmethod
    def random(shape):
        return np.random.randn(*shape), np.random.randn(shape[0], 1)

    @staticmethod
    def xavier_uni(shape):
        limit = np.sqrt(6 / (shape[0] + shape[1]))

        # TODO should anything be done with b(np.zeros...)?
        return np.random.uniform(
            low=-limit,
            high=limit,
            size=shape
        ), np.zeros(shape=(shape[0], 1))

    @staticmethod
    def he_uni(shape):
        limit = np.sqrt(6 / shape[1])

        return np.random.uniform(
            low=-limit,
            high=limit,
            size=shape
        ), np.zeros(shape=(shape[0], 1))

    @staticmethod
    def xavier_norm(shape):
        return np.random.normal(
            loc=0,
            scale=np.sqrt(2 / (shape[0] + shape[1])),
            size=shape
        ), np.zeros(shape=(shape[0], 1))

    @staticmethod
    def he_norm(shape):
        return np.random.normal(
            loc=0,
            scale=np.sqrt(2 / shape[1]),
            size=shape
        ), np.zeros(shape=(shape[0], 1))
