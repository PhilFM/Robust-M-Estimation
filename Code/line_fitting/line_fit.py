import math
import numpy as np

# Line model is y = a*x + b
class LineFit:
    def __init__(self):
        pass

    # r = a*xi + b - yi
    def residual(self, model, data_item, model_ref=None) -> np.array:
        a = model[0]
        b = model[1]
        x = data_item[0]
        y = data_item[1]
        return np.array([a*x + b - y])

    # dr/d(a b) = (x 1)
    def residual_gradient(self, model, data_item, model_ref=None) -> np.array:
        x = data_item[0]
        return np.array([[x, 1.0]])

    # return size of model if the model is linear, otherwise return 0
    def linear_model_size(self) -> int:
        return 2 # a,b
