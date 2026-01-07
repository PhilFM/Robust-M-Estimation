import numpy as np
import numpy.typing as npt

# Import cython helper.
from line_fit_orthog_welsch_fast import line_fit_orthog_welsch_objective_func, line_fit_orthog_welsch_update_weights
from line_fit_orthog_weighted_fit_fast import line_fit_orthog_weighted_fit_sums

class LineFitOrthogWelschEvaluator:
    def __init__(self):
        pass

    def set_residual_size(self, residual_size: npt.ArrayLike) -> None:
        residual_size[0] = 1

    def objective_func(self, model: npt.ArrayLike, model_ref, influence_func_instance,
                       data: npt.ArrayLike, weight: npt.ArrayLike, scale: npt.ArrayLike) -> float:
        return line_fit_orthog_welsch_objective_func(influence_func_instance.sigma,
                                              model[0], model[1], model[2], data[0], weight[0], scale[0])

    def update_weights(self, model: npt.ArrayLike, model_ref, influence_func_instance,
                       data: npt.ArrayLike, weight: npt.ArrayLike, scale: npt.ArrayLike,
                       new_weight: npt.ArrayLike) -> None:
        return line_fit_orthog_welsch_update_weights(influence_func_instance.sigma,
                                                     model[0], model[1], model[2], data[0], weight[0], scale[0], new_weight[0])

    def weighted_fit(self, data: npt.ArrayLike, weight: npt.ArrayLike, scale: npt.ArrayLike) -> np.array:
        X0 = np.zeros(2)
        cov = np.zeros((2,2))
        line_fit_orthog_weighted_fit_sums(data[0], weight[0], scale[0], X0, cov)

        # calculate normal vector as smallest eigenvalue of the covariance matrix
        e_val, e_vect = np.linalg.eig(cov)
        min_eval = np.argmin(e_val)
        normal_vector = e_vect[:, min_eval]
        return np.array([normal_vector[0], normal_vector[1], -np.dot(normal_vector,X0)]),None

