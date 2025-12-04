import numpy as np

class TRS:
    def __init__(self):
        pass

    def cache_model(self, model, model_ref=None):
        self.__s = model[0]
        self.__c = model[1]
        self.__tx = model[2]
        self.__ty = model[3]

    def residual(self, data_item) -> np.array:
        x1 = data_item[0]
        y1 = data_item[1]
        return np.array([data_item[2] - self.__c*x1 + self.__s*y1 - self.__tx,  # xdiff
                         data_item[3] - self.__s*x1 - self.__c*y1 - self.__ty]) # ydiff


    # d(xdiffi)/d(s c tx ty) = ( yi -xi -1  0)
    #  (ydiffi)                (-xi -yi  0 -1)
    def residual_gradient(self, data_item) -> np.array:
        x1 = data_item[0]
        y1 = data_item[1]
        return np.array([[ y1, -x1, -1.0,  0.0],
                         [-x1, -y1,  0.0, -1.0]])

    # return size of model if the model is linear, otherwise return 0
    def linear_model_size(self) -> int:
        return 4
