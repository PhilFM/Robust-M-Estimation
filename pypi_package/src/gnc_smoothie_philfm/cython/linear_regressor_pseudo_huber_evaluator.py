import numpy as np
import numpy.typing as npt

# Import cython helper.
try:
    from .linear_regressor_pseudo_huber_fast import linear_regressor_pseudo_huber_objective_func, linear_regressor_pseudo_huber_weighted_derivs, linear_regressor_pseudo_huber_update_weights
    from .linear_regressor_weighted_fit import linear_regressor_weighted_fit
except:
    from linear_regressor_pseudo_huber_fast import linear_regressor_pseudo_huber_objective_func, linear_regressor_pseudo_huber_weighted_derivs, linear_regressor_pseudo_huber_update_weights
    from linear_regressor_weighted_fit import linear_regressor_weighted_fit

class LinearRegressorPseudoHuberEvaluator:
    def __init__(self, data_item):
        if data_item.ndim == 2:
            self.__rsize = len(data_item)
            self.__msize = len(data_item[0])
        else:
            assert(data_item.ndim == 1)
            self.__rsize = 1
            self.__msize = len(data_item)

    def set_residual_size(self, residual_size: npt.ArrayLike) -> None:
        residual_size[0] = self.__rsize

    def objective_func(self, model: npt.ArrayLike, model_ref, influence_func_instance,
                       data: npt.ArrayLike, weight: npt.ArrayLike, scale: npt.ArrayLike) -> float:
        residual = np.zeros(self.__rsize)
        return linear_regressor_pseudo_huber_objective_func(influence_func_instance.sigma,
                                                            model, np.reshape(data[0], (len(data[0]), self.__rsize, self.__msize)), weight[0], scale[0], residual)

    def weighted_derivs(self, model: npt.ArrayLike, model_ref, influence_func_instance, lambda_val: float,
                        data: npt.ArrayLike, weight: npt.ArrayLike, scale: npt.ArrayLike) -> (np.array, np.array):
        residual = np.zeros(self.__rsize)
        grad = np.zeros((self.__rsize,self.__msize))
        a = np.zeros(self.__rsize*self.__msize)
        AlB = np.zeros((self.__rsize*self.__msize,self.__rsize*self.__msize))
        linear_regressor_pseudo_huber_weighted_derivs(influence_func_instance.sigma, lambda_val, model,
                                                      np.reshape(data[0], (len(data[0]), self.__rsize, self.__msize)),
                                                      weight[0], scale[0], residual, grad, a, AlB)
        return a, AlB

    def update_weights(self, model: npt.ArrayLike, model_ref, influence_func_instance,
                       data: npt.ArrayLike, weight: npt.ArrayLike, scale: npt.ArrayLike,
                       new_weight: npt.ArrayLike) -> None:
        residual = np.zeros(self.__rsize)
        return linear_regressor_pseudo_huber_update_weights(influence_func_instance.sigma, model,
                                                            np.reshape(data[0], (len(data[0]), self.__rsize, self.__msize)),
                                                            weight[0], scale[0], residual, new_weight[0])

    def weighted_fit(self, data: npt.ArrayLike, weight: npt.ArrayLike, scale: npt.ArrayLike=None) -> np.array:
        if scale is None:
            this_scale = np.zeros(len(data[0]))
            this_scale[:] = 1.0
        else:
            this_scale = scale[0]

        atot = np.zeros((self.__rsize,self.__msize))
        Atot = np.zeros((self.__rsize,self.__msize,self.__msize))
        linear_regressor_weighted_fit(np.reshape(data[0], (len(data[0]), self.__rsize, self.__msize)), weight[0], this_scale, atot, Atot)
        model = np.zeros(self.__rsize*self.__msize)
        for i in range(self.__rsize):
            model[i*self.__msize:(i+1)*self.__msize] = np.linalg.solve(Atot[i], atot[i])

        return model,None
