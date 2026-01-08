import numpy as np
import numpy.typing as npt

# Import cython helper.
try:
    from .linear_regressor_welsch_fast import linear_regressor_welsch_objective_func, linear_regressor_welsch_weighted_derivs, linear_regressor_welsch_update_weights
    from .linear_regressor_evaluator_base import LinearRegressorEvaluatorBase
except:
    from linear_regressor_welsch_fast import linear_regressor_welsch_objective_func, linear_regressor_welsch_weighted_derivs, linear_regressor_welsch_update_weights
    from linear_regressor_evaluator_base import LinearRegressorEvaluatorBase

class LinearRegressorWelschEvaluator(LinearRegressorEvaluatorBase):
    def __init__(self, data_item):
        LinearRegressorEvaluatorBase.__init__(self, data_item)

    def objective_func(self, model: npt.ArrayLike, model_ref, influence_func_instance,
                       data: npt.ArrayLike, weight: npt.ArrayLike, scale: npt.ArrayLike) -> float:
        residual = np.zeros(self._rsize)
        return linear_regressor_welsch_objective_func(influence_func_instance.sigma,
                                                      model, np.reshape(data[0], (len(data[0]), self._rsize, self._msize)), weight[0], scale[0], residual)

    def weighted_derivs(self, model: npt.ArrayLike, model_ref, influence_func_instance, lambda_b: float,
                        data: npt.ArrayLike, weight: npt.ArrayLike, scale: npt.ArrayLike) -> (np.array, np.array):
        residual = np.zeros(self._rsize)
        grad = np.zeros((self._rsize,self._msize))
        a = np.zeros(self._rsize*self._msize)
        AlB = np.zeros((self._rsize*self._msize,self._rsize*self._msize))
        linear_regressor_welsch_weighted_derivs(influence_func_instance.sigma, lambda_b, model,
                                                np.reshape(data[0], (len(data[0]), self._rsize, self._msize)),
                                                weight[0], scale[0], residual, grad, a, AlB)
        return a, AlB

    def update_weights(self, model: npt.ArrayLike, model_ref, influence_func_instance,
                       data: npt.ArrayLike, weight: npt.ArrayLike, scale: npt.ArrayLike,
                       new_weight: npt.ArrayLike) -> None:
        residual = np.zeros(self._rsize)
        return linear_regressor_welsch_update_weights(influence_func_instance.sigma, model,
                                                      np.reshape(data[0], (len(data[0]), self._rsize, self._msize)),
                                                      weight[0], scale[0], residual, new_weight[0])
