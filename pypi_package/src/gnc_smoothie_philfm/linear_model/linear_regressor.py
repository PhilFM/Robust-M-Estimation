import numpy as np

# Linear model is z = A*x + b
# Model is represented by matrix (A | b), observations by data items (x^T z)
class LinearRegressor:
    def __init__(self, data_item):
        if data_item.ndim == 2:
            self.__rsize = len(data_item)
            self.__msize = len(data_item[0])
        else:
            assert(data_item.ndim == 1)
            self.__rsize = 1
            self.__msize = len(data_item)

    # copy model parameters and apply any internal calculations
    def cache_model(self, model, model_ref=None):
        self.__model = np.copy(model)

    # r = coeff*x + intercept - yi
    def residual(self, data_item) -> np.array:
        resid = np.zeros(self.__rsize)
        dim = self.__msize-1
        if data_item.ndim == 2:
            for j in range(self.__rsize):
                for k in range(dim):
                    resid[j] += self.__model[j*self.__msize+k]*data_item[j][k]

                resid[j] += self.__model[j*self.__msize+dim] - data_item[j][dim]
        else:
            assert(data_item.ndim == 1 and self.__rsize == 1)
            for k in range(dim):
                resid[0] += self.__model[k]*data_item[k]

            resid[0] += self.__model[dim].item() - data_item[dim]

        return resid

    # dr/d(model) = (x y 1)
    def residual_gradient(self, data_item) -> np.array:
        resid_grad = np.zeros((self.__rsize,self.__rsize*self.__msize))
        dim = self.__msize-1
        if data_item.ndim == 2:
            for j in range(self.__rsize):
                for k in range(dim):
                    resid_grad[j][j*self.__msize+k] = data_item[j][k]

                resid_grad[j][j*self.__msize+dim] = 1.0
        else:
            assert(data_item.ndim == 1)
            for k in range(dim):
                resid_grad[0][k] = data_item[k]

            resid_grad[0][dim] = 1.0

        return resid_grad

    # return size of model if the model is linear, otherwise return 0
    def linear_model_size(self) -> int:
        return self.__rsize*self.__msize
