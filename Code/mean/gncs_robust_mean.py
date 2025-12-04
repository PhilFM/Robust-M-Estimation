import numpy as np

class RobustMean:
    def __init__(self):
        pass

    # copy model parameters and apply any internal calculations
    def cache_model(self, model, model_ref=None):
        self.__m = model[0]

    # r = m - d
    def residual(self, data_item, data_id=0) -> np.array:
        return np.array([self.__m-data_item[0]]).reshape(1)

    # dr/dm = 1
    def residual_gradient(self, data_item, data_id=0) -> np.array:
        return np.array([[1.0]])

    # return size of model if the model is linear, otherwise return 0
    def linear_model_size(self) -> int:
        return 1
