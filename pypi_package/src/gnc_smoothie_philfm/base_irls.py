# Base class for IRLS and SupGaussNewton classes, contains stuff common to both. Not to be used by itself.
import numpy as np


class BaseIRLS:
    def __init__(
        self,
        param_instance,
        model_instance,
        data,
        data_ids=None,
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
        self._param_instance = param_instance
        self._model_instance = model_instance
        self._data = data
        if data_ids is None:
            self._data_ids = np.zeros(len(data), dtype=int)
        else:
            if len(data_ids) != len(data):
                raise ValueError("Inconsistent data ID array")

            self._data_ids = data_ids

        if weight is None:
            self._weight = np.zeros(len(data))
            self._weight[:] = 1.0
        else:
            if len(weight) != len(data):
                raise ValueError("Inconsistent weight array")

            self._weight = weight

        if scale is None:
            self._scale = np.zeros(len(data))
            self._scale[:] = 1.0
        else:
            if len(scale) != len(data):
                raise ValueError("Inconsistent scale array")

            for s in scale:
                if s < 1.0:
                    raise ValueError("Scale value less than one")

            self._scale = scale

        self.numeric_derivs_model = numeric_derivs_model
        self.numeric_derivs_influence = numeric_derivs_influence

        # GNC schedule for sigma
        if self._param_instance.n_steps() > max_niterations:
            raise ValueError("Too many GNC steps relative to # iterations")

        self._max_niterations = max_niterations
        self._diff_thres = diff_thres
        self._print_warnings = print_warnings
        self._model_start = model_start
        self._model_ref_start = model_ref_start
        self._debug = debug
        self._linear_model_size = getattr(model_instance, "linear_model_size", None)
        
    def objective_func_sign(self) -> float:
        return self._param_instance.influence_func_instance.objective_func_sign()

    # objective_func() is public to allow external checking of progress
    def objective_func(self, model, weight=None, model_ref=None) -> float:
        if weight is None:
            weight = self._weight

        tot = 0.0
        self._model_instance.cache_model(model, model_ref=model_ref)
        for d, di, w, s in zip(self._data, self._data_ids, weight, self._scale, strict=True):
            residual = self._model_instance.residual(d, di)
            tot += w * self._param_instance.influence_func_instance.rho(
                residual @ residual, s
            )  # scale

        self._residual_size = len(residual)
        return tot

    def _init_model(self) -> None:
        if self._model_start is None and self._model_ref_start is None:
            if callable(self._linear_model_size):
                return self.weighted_fit(), None
            else:
                return self._model_instance.weighted_fit(
                    self._data, self._data_ids, self._weight, self._scale
                )
        else:
            return self._model_start, self._model_ref_start

    def _calc_residual_derivatives(self, model, model_ref = None, small_diff = 1.0e-5) -> (np.array, np.array):
        # build arrays of residuals and gradients per data item
        residual_arr = np.zeros((len(self._data), self._residual_size))
        residual_gradient_arr = np.zeros((len(self._data), self._residual_size, len(model)))

        # first the residuals
        self._model_instance.cache_model(model, model_ref=model_ref)
        for i,(d,di) in enumerate(zip(self._data, self._data_ids, strict=True)):
            residual = self._model_instance.residual(d, di)
            residual_arr[i] = residual

        # now the gradients
        if self.numeric_derivs_model:
            model_copy = np.copy(model)
            for i in range(len(model)):
                model_copy[i] -= small_diff
                self._model_instance.cache_model(model_copy, model_ref=model_ref)
                residual_n_arr = np.zeros((len(self._data), self._residual_size))
                for j,(d,di) in enumerate(zip(self._data, self._data_ids, strict=True)):
                    residual = self._model_instance.residual(d, di)
                    residual_n_arr[j] = residual

                model_copy[i] += 2.0 * small_diff
                self._model_instance.cache_model(model_copy, model_ref=model_ref)
                for j,(d,di) in enumerate(zip(self._data, self._data_ids, strict=True)):
                    residual = self._model_instance.residual(d, di)
                    for k in range(self._residual_size):
                        residual_gradient_arr[(j,k,i)] = 0.5*(residual[k] - residual_n_arr[j][k]) / small_diff
                    
                model_copy[i] = model[i]
        else:
            for j,(d,di) in enumerate(zip(self._data, self._data_ids, strict=True)):
                residual_gradient_arr[j] = self._model_instance.residual_gradient(d, di)

        return residual_arr, residual_gradient_arr

    def weighted_fit(self, weight=None) -> np.array:
        if weight is None:
            weight = self._weight

        model = np.zeros(self._linear_model_size())

        # initialize residual_size
        self._model_instance.cache_model(model)
        residual = self._model_instance.residual(self._data[0], self._data_ids[0])
        self._residual_size = len(residual)

        atot = np.zeros(len(model))
        Atot = np.zeros((len(model), len(model)))
        small_diff = 1.0e-5  # in case numerical differentiation is specified
        residual_arr, residual_gradient_arr = self._calc_residual_derivatives(model, small_diff)
        for i,(w,s) in enumerate(zip(weight, self._scale, strict=True)):
            grad = np.matmul(np.transpose(residual_gradient_arr[i]), residual_arr[i])
            w /= s * s
            atot += w * grad
            Atot += w * np.matmul(
                np.transpose(residual_gradient_arr[i]), residual_gradient_arr[i]
            )

        return -np.linalg.solve(Atot, atot)
