import numpy as np

# Plane model is z = a*x + b*y + c
class PlaneFit:
    def __init__(self):
        pass

    # copy model parameters and apply any internal calculations
    def cache_model(self, model, model_ref=None):
        self.__a = model[0]
        self.__b = model[1]
        self.__c = model[2]

    # r = a*xi + b - yi
    def residual(self, data_item) -> np.array:
        x = data_item[0]
        y = data_item[1]
        z = data_item[2]
        return np.array([self.__a*x + self.__b*y + self.__c - z])

    # dr/d(a b c) = (x y 1)
    def residual_gradient(self, data_item) -> np.array:
        x = data_item[0]
        y = data_item[1]
        return np.array([[x, y, 1.0]])

    # return size of model if the model is linear, otherwise return 0
    def linear_model_size(self) -> int:
        return 3 # a,b,c
