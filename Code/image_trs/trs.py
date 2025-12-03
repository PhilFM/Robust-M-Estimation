import numpy as np
import math

class TRS:
    def __init__(self):
        pass

    def residual(self, model, data_item, model_ref=None) -> np.array:
        s = model[0]
        c = model[1]
        tx = model[2]
        ty = model[3]
        x1 = data_item[0]
        y1 = data_item[1]
        return np.array([data_item[2] - c*x1 + s*y1 - tx,  # xdiff
                         data_item[3] - s*x1 - c*y1 - ty]) # ydiff


    # d(xdiffi)/d(s c tx ty) = ( yi -xi -1  0)
    #  (ydiffi)                (-xi -yi  0 -1)
    def residual_gradient(self, model, data_item, model_ref=None) -> np.array:
        x1 = data_item[0]
        y1 = data_item[1]
        return np.array([[ y1, -x1, -1.0,  0.0],
                         [-x1, -y1,  0.0, -1.0]])

    # return size of model if the model is linear, otherwise return 0
    def linear_model_size(self) -> int:
        return 4
