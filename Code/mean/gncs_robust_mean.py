import numpy as np
import math

class RobustMean:
    def __init__(self):
        pass

    def residual(self, model, data_item, model_ref=None) -> np.array:
        return np.array([model[0]-data_item[0]]).reshape(1)

    def residual_gradient(self, model, data_item, model_ref=None) -> np.array:
        return np.array([[1.0]])

    # return size of model if the model is linear, otherwise return 0
    def linear_model_size(self) -> int:
        return 1
