import numpy as np

# Line model is y = a*x + b
class LineFit:
    def __init__(self):
        pass

    # copy model parameters and apply any internal calculations
    def cache_model(self, model, model_ref=None):
        self.__a = model[0]
        self.__b = model[1]

    # r = a*xi + b - yi
    def residual(self, data_item) -> np.array:
        x = data_item[0]
        y = data_item[1]
        return np.array([self.__a*x + self.__b - y])

    # dr/d(a b) = (x 1)
    def residual_gradient(self, data_item) -> np.array:
        x = data_item[0]
        return np.array([[x, 1.0]])

    # return size of model if the model is linear, otherwise return 0
    def linear_model_size(self) -> int:
        return 2 # a,b
