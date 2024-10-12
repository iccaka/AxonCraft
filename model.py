import sys
import numpy as np

from typing import Sequence
from tqdm import trange
from emptymodelerror import EmptyModelError
from layer import Layer
from nolossfunctionerror import NoLossFunctionError

class Model:
    # TODO add add/remove/replace layer functionality(e.g. when an empty array is passed at first)
    # TODO add learning rate decay
    # TODO validation for optimizer and cost
    def __init__(self, layers: Sequence[Layer] = []):
        # TODO __cache shouldn't be a field (maybe idk)
        self._optimizer = None
        self._loss = None
        self._learning_rate = None
        self._batch_norm = None
        self._grad = None
        # TODO could probably be made like it is in tf -> dict history -> returned after calling fit()
        self.__layers = np.array(layers)
        self.__cache = None
        self.__are_weights_initialized = False
        self.__are_BN_parameters_initialized = False

        self._update_layer_names()

    def summary(self):
        if len(self.__layers) == 0:
            print('This model doesn\'t have any layers and thus there\'s nothing to be shown.')
            return

        if self.__are_weights_initialized:
            weights = self.get_weights()
            total_param_count = 0

            for i in range(0, len(weights), 2):
                layer = self.__layers[int(i / 2)]
                total_param_count += layer.param_count

                print('Name: {} / Units: {} / Activation: {} / Initializer: {}\n'
                      '\t# of params: {}\n'
                      '\tw: {} / b: {}'.format(
                    layer.name,
                    layer.unit_count,
                    layer.activation.__name__,
                    layer.initializer.__name__,
                    layer.param_count,
                    weights[i].shape,
                    weights[i + 1].shape
                ))

            print('Total params: {}'.format(total_param_count))
        else:
            for layer in self.__layers:
                print('Name: {} / Units: {} / Activation: {} / Initializer: {}'.format(
                    layer.name,
                    layer.unit_count,
                    layer.activation.__name__,
                    layer._initializer.__name__
                ))

    def get_layer(self, name=None, position=None) -> Layer:
        if len(self.__layers) == 0:
            raise EmptyModelError('The model doesn\'t have any layers.')

        if position is None and name is None:
            raise ValueError('You must pass either a name or a position of the layer.')

        if position is not None:
            try:
                return self.__layers[position]
            except Exception:
                raise ValueError('No layer found at position: {}'.format(str(position)))
        else:
            for layer in self.__layers:
                if layer._name == name:
                    return layer

            raise ValueError('No layer with such name found: {}'.format(name))

    # TODO maybe do it with @property?
    def get_weights(self):
        if not self.__are_weights_initialized:
            raise ValueError('Weights are not initialized. To do so run either fit() or build().')

        weights = []

        for layer in self.__layers:
            layer_weights = layer.get_weights()
            weights.append(layer_weights[0])
            weights.append(layer_weights[1])

        return weights

    def get_BN_parameters(self):
        if not self.__are_BN_parameters_initialized:
            raise ValueError('BN parameters are not initialized. To do so run either fit() or build() '
                             'after using configure() and passing batch_norm=True.')

        params = []

        for layer in self.__layers:
            layer_BN_params = layer.get_BN_parameters()
            params.append(layer_BN_params[0])
            params.append(layer_BN_params[1])

            return params

    def set_weights(self, weights):
        if self.__are_weights_initialized:
            current_weights = self.get_weights()

            if len(weights) != len(current_weights):
                raise ValueError('The number of weights provided doesn\'t match with the current ones.\n'
                                 'Expected: {}\n'
                                 'Provided: {}'.format(
                    len(current_weights),
                    len(weights)
                ))

            for i, weight in enumerate(current_weights):
                if weight.shape != weights[i].shape:
                    raise ValueError('Weights\' shapes for 1 or more of them don\'t match.')

        for i in range(0, len(weights), 2):
            w = weights[i]
            b = weights[i + 1]
            self.__layers[int(i / 2)].set_weights(w, b)

    # TODO maybe treat layer 0 like a Layer
    def build(self, _input_shape=None):
        if len(self.__layers) == 0:
            raise EmptyModelError('The model cannot be built because no layers have been added.')

        if self.__layers[0].input_shape is None:
            if _input_shape is None:
                raise ValueError('You must specify the input shape for the first layer.')
            else:
                self.__layers[0].input_shape = _input_shape

        for i, layer in enumerate(self.__layers):
            layer.set_weights(*layer.initializer(
                shape=(layer.unit_count, self.__layers[i - 1].unit_count) if i != 0
                else (layer.unit_count, layer.input_shape[1])
            ))

        self.__are_weights_initialized = True

        if self.batch_norm and not self.__are_BN_parameters_initialized:
            self.__initialize_BN_params()

    # TODO check x and y's shapes
    # TODO add batch size functionality
    def fit(self, X, y, epochs):
        if len(self.__layers) == 0:
            raise EmptyModelError('The model cannot be fit because no layers have been added.')

        if self._loss is None:
            raise NoLossFunctionError('The model cannot be fit because there\'s no loss function chosen. '
                                      'To choose one, use configure().')

        if not self.__are_weights_initialized:
            self.build(_input_shape=X.shape)
        else:
            expected = self.__layers[0].get_weights()[0].shape[1]

            if X.shape[1] != expected:
                raise ValueError('Training data\'s shape doesn\'t match that of the 1st layer\'s weights\' shape.\n'
                                 'Expected: (x, {})\n'
                                 'Provided: {}, where \'x\' = training examples.'.format(
                    expected,
                    X.shape
                ))

        if self.batch_norm and not self.__are_BN_parameters_initialized:
            self.__initialize_BN_params()

        cost_cache = []

        for _ in trange(epochs, desc='Training...', file=sys.stdout):
            self.__cache = []

            predictions = self._forward_prop(X)
            cost_cache.append(self._loss(predictions, y))
            # dA = self._grad(X, predictions, y)
            dA = predictions - y.reshape(-1, 1)
            self._update_weights(dA, X)

        print('Training complete!')

        return cost_cache

    # TODO finish configure(equal to tf's compile)
    def configure(self,
                  loss,
                  optimizer: str = 'rmsprop',
                  learning_rate: float = 0.01,
                  batch_norm: bool = False
                  ):
        if loss is None:
            raise ValueError('The loss cannot be empty.')

        # TODO maybe move these methods somewhere else
        if loss == 'categorical_crossentropy':
            self._loss = self.categorical_crossentropy
            self._grad = self._categorical_crossentropy_gradient
        elif loss == 'sparse_categorical_crossentropy':
            self._loss = self.sparse_categorical_crossentropy
            self._grad = self._sparse_categorical_crossentropy_gradient
        elif loss == 'binary_crossentropy':
            self._loss = self.binary_crossentropy
            self._grad = self._binary_crossentropy_gradient
        elif loss == 'mean_squared_error':
            self._loss = self.mean_squared_error
            self._grad = self._mean_squared_error_gradient
        elif loss == 'mean_absolute_error':
            self._loss = self.mean_absolute_error
            self._grad = self._mean_absolute_error_gradient
        else:
            raise ValueError('No such loss function.')

        # TODO maybe make it like it is in tf -> optimizers.get ...
        if optimizer == 'rmsprop':
            pass
        elif optimizer == 'gd':
            pass
        elif optimizer == 'adam':
            pass
        elif optimizer == 'sgd':
            pass

        self._learning_rate = learning_rate
        self._batch_norm = batch_norm

    # TODO finish evaluate
    def evaluate(self):
        pass

    # TODO finish predict
    def predict(self):
        pass

    def __initialize_BN_params(self):
        for i, layer in enumerate(self.__layers):
            layer.set_BN_parameters(
                np.full(shape=(layer.unit_count, 1), fill_value=1, dtype=float),
                np.zeros(shape=(layer.unit_count, 1))
            )

        self.__are_BN_parameters_initialized = True

    # TODO vectorized
    @staticmethod
    def categorical_crossentropy(y, y_hat):
        # for when labels are one-hot encoded

        return (-1 / y.shape[0]) * np.sum(y_hat * np.log(y) + (1 - y_hat) * np.log(1 - y))

    @staticmethod
    def sparse_categorical_crossentropy(predictions, y):
        # for when the labels are integers representing class indices

        return np.mean(-np.log(predictions[np.arange(y.shape[0]), y]))
        # return -np.log(predictions[y])

    # or just cross-entropy
    @staticmethod
    def binary_crossentropy():
        pass

    @staticmethod
    def mean_squared_error(predictions, y):
        return (1 / y.shape[0]) * np.sum((y - predictions) ** 2)

    @staticmethod
    def mean_absolute_error():
        pass

    @staticmethod
    def _categorical_crossentropy_gradient():
        pass

    @staticmethod
    def _sparse_categorical_crossentropy_gradient(predictions, y_true):
        # list comprehension method
        # return [
        #     [predictions[i][j] - 1 if j == y_true[i] else predictions[i][j] for j, _ in enumerate(example)]
        #         for i, example in enumerate(predictions)
        # ]

        result = np.copy(predictions)
        result[np.arange(predictions.shape[0]), y_true] -= 1

        return result

    @staticmethod
    def _binary_crossentropy_gradient():
        pass

    @staticmethod
    def _mean_squared_error_gradient(X, predictions, y_true):
        return (1 / y_true.shape[0]) * np.sum(np.matmul((y_true - predictions), X))

    @staticmethod
    def _mean_absolute_error_gradient():
        pass

    def _update_layer_names(self):
        for i, layer in enumerate(self.__layers):
            if layer.name == '_':
                layer.name = 'layer_{}'.format(str(i + 1))

    def _forward_prop(self, input):
        A = input

        for layer in self.__layers:
            layer_W, layer_b = layer.get_weights()
            Z = Layer.linear_transform(layer_W, layer_b, A)
            A = layer.activation(Z)
            self.__cache.append([A, Z, layer_W, layer_b])

        return A

    def _update_weights(self, dA, X):
        m = X.shape[0]
        A = dA

        for i, cache in reversed(list(enumerate(self.__cache))):
            curr_layer = self.get_layer(position=i)

            dZ = A * curr_layer.activation_grad(cache[1])
            dW = (1 / m) * np.dot(dZ.T, self.__cache[i - 1][0] if (i - 1) != -1 else X)
            db = (1 / m) * np.sum(dZ, axis=0, keepdims=True)
            dA = np.dot(dZ, cache[2])

            curr_layer.set_weights(
                cache[2] - (self._learning_rate * dW),
                cache[3] - (self._learning_rate * db.T)
            )

            A = dA

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def loss(self):
        return self._loss

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def batch_norm(self):
        return self._batch_norm
