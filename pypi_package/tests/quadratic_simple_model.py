import numpy as np

# Test class for second derivatives in IRLS models.
# Implements model r = [a^2 - a*b - 3*b^2 + a*b*c - x,
#                       3.4*a*b^2 - c^3 - y] where a,b are the model parameter and x,y are the data values
class QuadraticSimpleModel:
    def __init__(self):
        pass

    # copy model parameters and apply any internal calculations
    def cache_model(self, model, model_ref=None):
        self.__model = np.copy(model)

    def residual(self, data_item) -> np.array:
        a = self.__model[0]
        b = self.__model[1]
        c = self.__model[2]
        x = data_item[0]
        y = data_item[1]
        return np.array([a*a - a*b - 3.0*b*b + a*b*c - x, 3.4*a*b*b - c*c*c - y])

    def residual_gradient(self, data_item) -> np.array:
        a = self.__model[0]
        b = self.__model[1]
        c = self.__model[2]
        x = data_item[0]
        y = data_item[1]
        return np.array([[2*a-b+b*c, -a-6.0*b+a*c, a*b],
                         [3.4*b*b, 6.8*a*b, -3.0*c*c]])

    def residual_2nd_deriv(self, data_item) -> np.array:
        a = self.__model[0]
        b = self.__model[1]
        c = self.__model[2]
        x = data_item[0]
        y = data_item[1]
        return np.array([[[ 2.0, -1.0+c, b],
                          [-1.0+c, -6.0, a],
                          [  b,  a, 0.0]],
                         [[ 0.0, 6.8*b, 0.0],
                          [6.8*b, 6.8*a, 0.0],
                          [0.0, 0.0, -6.0*c]]])
