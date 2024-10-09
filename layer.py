import numpy as np

from typing import List
from initializer import Initializer

class Layer:
    def __init__(self,
                 unit_count: int,
                 input_shape: tuple = None,
                 activation: str = 'linear',
                 name='_',
                 initializer: Initializer = Initializer.random):
        self.__are_weights_initialized = False

        self.unit_count = unit_count
        self.input_shape = input_shape

        # TODO keep in mind there's a difference between self and Layer. ... here
        # TODO use dict? (activations.get?)
        if activation == 'linear':
            self.activation = self.linear
            self.activation_grad = self.linear_gradient
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
            self.activation_grad = self.sigmoid_gradient
        elif activation == 'tanh':
            self.activation = self.tanh
            self.activation_grad = self.tanh_gradient
        elif activation == 'softmax':
            self.activation = self.softmax
            self.activation_grad = self.softmax_gradient
        elif activation == 'relu':
            self.activation = self.relu
            self.activation_grad = self.relu_gradient
        elif activation == 'leaky_relu':
            self.activation = self.leaky_relu
            self.activation_grad = self.leaky_relu_gradient
        else:
            raise ValueError('No such activation function.')

        self.name = name
        self.initializer = initializer
        self.param_count = 0

        self.__W = None
        self.__b = None

    # TODO move function definitions to a separate class/file?
    # TODO derivative calculation in a separate class/file?
    # TODO add PReLU
    @staticmethod
    def linear_transform(W, b, A):
        return np.matmul(A, W.T) + b.T

    @staticmethod
    def linear(Z):
        return Z

    @staticmethod
    def sigmoid(Z: np.ndarray) -> np.ndarray:
        if not isinstance(Z, np.ndarray):
            raise TypeError('Input must be a numpy array. Instead got {}.'.format(type(Z)))

        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def tanh(Z: np.ndarray) -> np.ndarray:
        if not isinstance(Z, np.ndarray):
            raise TypeError('Input must be a numpy array. Instead got {}.'.format(type(Z)))

        e_z = np.exp(Z)
        me_z = np.exp(-Z)

        return (e_z - me_z) / (e_z + me_z)

    @staticmethod
    def softmax(Z) -> np.ndarray:
        e_z = np.exp(Z - Z.max())

        return e_z / np.sum(e_z)

    @staticmethod
    def relu(Z) -> np.ndarray:
        return np.maximum(0, Z)

    # TODO Maybe do something about the alpha(a) values both here and in leaky_relu_gradient().
    @staticmethod
    def leaky_relu(Z: np.ndarray, a=0.01) -> np.ndarray:
        if not isinstance(Z, np.ndarray):
            raise TypeError('Input must be a numpy array. Instead got {}.'.format(type(Z)))

        return np.maximum(a * Z, Z)

    @staticmethod
    def linear_gradient(Z: np.ndarray) -> np.ndarray:
        if not isinstance(Z, np.ndarray):
            raise TypeError('Input must be a numpy array. Instead got {}.'.format(type(Z)))

        return np.full(Z.shape, 1, dtype=int)

    @staticmethod
    def sigmoid_gradient(Z: np.ndarray) -> np.ndarray:
        s = Layer.sigmoid(Z)

        return s * (1 - s)

    # TODO Should it be double checked(once already in Layer.tanh) if the type is np.ndarray?
    @staticmethod
    def tanh_gradient(Z: np.ndarray) -> np.ndarray:
        return 1 - (Layer.tanh(Z) ** 2)

    @staticmethod
    def softmax_gradient(Z) -> np.ndarray:
        pass


    @staticmethod
    def relu_gradient(Z: np.ndarray) -> np.ndarray:
        if not isinstance(Z, np.ndarray):
            raise TypeError('Input must be a numpy array. Instead got {}.'.format(type(Z)))

        return np.where(Z <= 0, 0, 1)

    # TODO Maybe do something about the alpha(a) values both here and in leaky_relu().
    @staticmethod
    def leaky_relu_gradient(Z: np.ndarray, a=0.01) -> np.ndarray:
        if not isinstance(Z, np.ndarray):
            raise TypeError('Input must be a numpy array. Instead got {}.'.format(type(Z)))

        return np.where(Z <= 0, a, 1)

    # TODO maybe do it with @property?
    def get_weights(self) -> tuple[np.ndarray, np.ndarray] | List:
        if not self.__are_weights_initialized:
            print('The weights are not yet initialized. '
                  'Please run either fit() or build() on your model before calling.')
            return []

        return self.__W, self.__b

    def set_weights(self, W: np.ndarray, b: np.ndarray):
        # TODO check whether this is the 1st layer and if everything matches
        if not isinstance(W, np.ndarray):
            raise TypeError('W must be of type numpy array. Instead got {}.'.format(type(W)))

        if not isinstance(b, np.ndarray):
            raise TypeError('b must be of type numpy array. Instead got {}.'.format(type(b)))

        if self.__are_weights_initialized:
            if self.__W.shape != W.shape or self.__b.shape != b.shape:
                raise ValueError('The provided weights\' shapes don\'t match with the existing ones.\n'
                                 'Expected: w: {} / b: {}\n'
                                 'Provided: w: {} / b: {}'.format(
                    self.__W.shape,
                    self.__b.shape,
                    W.shape,
                    b.shape
                )
                )
        else:
            self.__are_weights_initialized = True

        self.__W, self.__b = W, b
        self.param_count = (W.shape[1] * self._unit_count) + b.shape[0]

    @property
    def input_shape(self) -> tuple:
        return self._input_shape

    @input_shape.setter
    def input_shape(self, new_value: tuple | None = None):
        if self.__are_weights_initialized:
            raise ValueError('You cannot change the input shape '
                             'after the weights have already been initialized.')

        if new_value is not None and not isinstance(new_value, tuple):
            raise TypeError('Input must be equal to None or be a tuple. Instead got {}.'.format(type(new_value)))

        self._input_shape = new_value

    @property
    def unit_count(self) -> int:
        return self._unit_count

    @unit_count.setter
    def unit_count(self, new_value: int):
        if not isinstance(new_value, int):
            raise TypeError('Input must be an int. Instead got {}.'.format(type(new_value)))

        if new_value <= 0:
            raise ValueError('The unit count must be a positive integer.')

        self._unit_count = new_value

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_value: str):
        if not new_value:
            raise ValueError('Names cannot be empty.')

        if not isinstance(new_value, str):
            raise TypeError('Input must be a string. Instead got {}.'.format(type(new_value)))

        self._name = new_value

    @property
    def initializer(self) -> Initializer:
        return self._initializer

    @initializer.setter
    def initializer(self, new_value: Initializer):
        if self.__are_weights_initialized:
            raise ValueError('You cannot change the initializer algorithm '
                             'after the weights have already been initialized.')

        self._initializer = new_value

    @property
    def param_count(self) -> int:
        return self._param_count

    @param_count.setter
    def param_count(self, new_value: int):
        if not isinstance(new_value, int):
            raise TypeError('Input must be an int. Instead got {}.'.format(type(new_value)))

        if new_value < 0:
            raise ValueError('The parameter count must be an int equal or greater than zero.')

        self._param_count = new_value
