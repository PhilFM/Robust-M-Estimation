# Base class for IRLS and SupGaussNewton classes, contains stuff common to both. Not to be used by itself.
import numpy as np


class BaseIRLS:
    def __init__(
        self,
        param_instance,
        model_instance,
        data,
        weight=None,
        scale=None,
        numeric_derivs_model: bool = False,
        numeric_derivs_influence: bool = False,
        max_niterations: int = 50,
        diff_thres: float = 1.0e-12,
        print_warnings: bool = False,
        model_start=None,
        model_ref_start=None,
        debug: bool = False,
    ):
        self.param_instance = param_instance
        self.model_instance = model_instance
        self.data = data
        self.weight = weight
        if weight is None:
            self.weight = np.zeros(len(data))
            self.weight[:] = 1.0
        else:
            if len(weight) != len(data):
                raise ValueError("Inconsistent weight array")

        self.scale = scale
        if scale is not None:
            if len(scale) != len(data):
                raise ValueError("Inconsistent scale array")

            for s in scale:
                if s < 1.0:
                    raise ValueError("Scale value less than one")

        self.numeric_derivs_model = numeric_derivs_model
        self.numeric_derivs_influence = numeric_derivs_influence
        self.max_niterations = max_niterations
        self.diff_thres = diff_thres
        self.print_warnings = print_warnings
        self.model_start = model_start
        self.model_ref_start = model_ref_start
        self.debug = debug

    def init_model(self):
        if self.model_start is None and self.model_ref_start is None:
            if self.model_instance.linear_model_size() > 0:
                return self.weighted_fit(), None
            else:
                return self.model_instance.weighted_fit(
                    self.data, self.weight, self.scale
                )
        else:
            return self.model_start, self.model_ref_start

    def objective_func(self, model, weight=None, model_ref=None) -> float:
        if weight is None:
            weight = self.weight

        tot = 0.0
        if self.scale is None:
            for d, w in zip(self.data, weight, strict=True):
                residual = self.model_instance.residual(model, d, model_ref)
                tot += w * self.param_instance.influence_func_instance.rho(
                    residual @ residual, 1.0
                )  # scale
        else:
            for d, w, s in zip(self.data, weight, self.scale, strict=True):
                residual = self.model_instance.residual(model, d, model_ref)
                tot += w * self.param_instance.influence_func_instance.rho(
                    residual @ residual, s
                )  # scale

        return tot

    def residual_gradient_numerical(
        self, model, d, residual_size: int, model_ref=None, small_diff: float = 1.0e-5
    ) -> np.array:
        residual_gradient = np.zeros((residual_size, len(model)))
        model_copy = np.copy(model)
        for i in range(len(model)):
            model_copy[i] -= small_diff
            residualn = self.model_instance.residual(model_copy, d, model_ref)
            model_copy[i] += 2.0 * small_diff
            residualp = self.model_instance.residual(model_copy, d, model_ref)
            model_copy[i] = model[i]
            for j in range(residual_size):
                residual_gradient[j][i] = (
                    0.5 * (residualp[j] - residualn[j]) / small_diff
                )

        return residual_gradient

    def calc_residual_derivatives(
        self, model, data_item, model_ref=None, small_diff: float = 1.0e-5
    ) -> (np.array, np.array, np.array):
        residual = self.model_instance.residual(model, data_item, model_ref)
        if self.numeric_derivs_model:
            residual_gradient = self.residual_gradient_numerical(
                model,
                data_item,
                residual.shape[0],
                model_ref=model_ref,
                small_diff=small_diff,
            )
        else:
            residual_gradient = self.model_instance.residual_gradient(
                model, data_item, model_ref=model_ref
            )

        grad = np.matmul(np.transpose(residual_gradient), residual)
        return residual, residual_gradient, grad

    def weighted_fit(self, weight=None) -> np.array:
        if weight is None:
            weight = self.weight

        model = np.zeros(self.model_instance.linear_model_size())
        atot = np.zeros(len(model))
        Atot = np.zeros((len(model), len(model)))
        small_diff = 1.0e-5  # in case numerical differentiation is specified
        if self.scale is None:
            for d, w in zip(self.data, weight, strict=True):
                residual, residual_gradient, grad = self.calc_residual_derivatives(
                    model, d, model_ref=None, small_diff=small_diff
                )
                atot += w * grad
                Atot += w * np.matmul(
                    np.transpose(residual_gradient), residual_gradient
                )
        else:
            for d, w, s in zip(self.data, weight, self.scale, strict=True):
                residual, residual_gradient, grad = self.calc_residual_derivatives(
                    model, d, model_ref=None, small_diff=small_diff
                )
                w /= s * s
                atot += w * grad
                Atot += w * np.matmul(
                    np.transpose(residual_gradient), residual_gradient
                )

        return -np.linalg.solve(Atot, atot)
