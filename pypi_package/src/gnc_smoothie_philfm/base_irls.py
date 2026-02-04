# Base class for IRLS and SupGaussNewton classes, contains stuff common to both. Not to be used by itself.
import numpy as np
import numpy.typing as npt
from typing import TextIO
import math


class BaseIRLS:
    # number of types of data supported
    _dsize = 3

    def __assign_data(
        self,
        didx: int,
        data, weight:
        npt.ArrayLike,
        scale: npt.ArrayLike
    ):
        if didx >= self._dsize:
            raise ValueError("Inconsistent data array index", didx)

        if data is not None:
            self._data[didx] = data
            if weight is None:
                self._weight[didx] = np.ones(len(data))
            else:
                if len(weight) != len(data):
                    raise ValueError("Inconsistent weight array", didx)

                self._weight[didx] = weight

            if scale is None:
                self._scale[didx] = np.zeros(len(data))
                self._scale[didx][:] = 1.0
            else:
                if len(scale) != len(data):
                    raise ValueError("Inconsistent scale array", didx)

                for s in scale:
                    if s < 1.0:
                        raise ValueError("Scale value less than one", didx)

                self._scale[didx] = scale

    def __init__(
        self,
        param_instance,
        data: npt.ArrayLike,
        *,
        model_instance = None, # Python model
        evaluator_instance = None, # Cython model
        weight: npt.ArrayLike = None,
        scale: npt.ArrayLike = None,
        data2=None,
        weight2: npt.ArrayLike = None,
        scale2: npt.ArrayLike = None,
        data3=None,
        weight3: npt.ArrayLike = None,
        scale3: npt.ArrayLike = None,
        numeric_derivs_model: bool = False,
        numeric_derivs_influence: bool = False,
        max_niterations: int = 50,
        diff_thres: float = 1.0e-12,
        messages_file: TextIO = None,
        model_start: npt.ArrayLike = None,
        model_ref_start: npt.ArrayLike = None,
        debug: bool = False,
    ):
        self._param_instance = param_instance
        self._model_instance = model_instance
        self._evaluator_instance = evaluator_instance

        self._data = [None] * self._dsize
        self._weight = [None] * self._dsize
        self._scale = [None] * self._dsize
        self.__assign_data(0, data, weight, scale)
        self.__assign_data(1, data2, weight2, scale2)
        self.__assign_data(2, data3, weight3, scale3)

        self.numeric_derivs_model = numeric_derivs_model
        self.numeric_derivs_influence = numeric_derivs_influence

        # GNC schedule for sigma
        if self._param_instance.n_steps() > max_niterations:
            raise ValueError("Too many GNC steps relative to # iterations")

        self._max_niterations = max_niterations
        self._diff_thres = diff_thres
        self._messages_file = messages_file
        self._model_start = None if model_start is None else np.copy(model_start)
        self._model_ref_start = None if model_ref_start is None else np.copy(model_ref_start)
        self._debug = debug
        self._linear_model_size = getattr(model_instance, "linear_model_size", None)

        self._residual_size = None

    def param_instance(self):
        return self._param_instance

    def model_instance(self):
        return self._model_instance

    def evaluator_instance(self):
        return self._evaluator_instance

    def objective_func_sign(self) -> float:
        return self._param_instance.influence_func_instance.objective_func_sign()

    def _get_model_residual_func(
            self,
            didx: int):
        if didx >= self._dsize:
            raise ValueError("Inconsistent data array index", didx)

        if self._data[didx] is not None:
            if didx == 0:
                return self._model_instance.residual
            elif didx == 1:
                return self._model_instance.residual2
            elif didx == 2:
                return self._model_instance.residual3
            else:
                raise ValueError("Inconsistent data array index", didx)

    def _get_model_residual_gradient_func(
            self,
            didx: int):
        if didx >= self._dsize:
            raise ValueError("Inconsistent data array index", didx)

        if self._data[didx] is not None:
            if didx == 0:
                return self._model_instance.residual_gradient
            elif didx == 1:
                return self._model_instance.residual_gradient2
            elif didx == 2:
                return self._model_instance.residual_gradient3
            else:
                raise ValueError("Inconsistent data array index", didx)

    # objective_func() is public to allow external checking of progress
    def objective_func(self,
                       model: npt.ArrayLike,
                       *,
                       model_ref=None) -> float:
        self._residual_size = [None] * self._dsize
        if self._evaluator_instance is None:
            tot = 0.0
            self._model_instance.cache_model(model, model_ref=model_ref)
            for didx in range(self._dsize):
                if self._data[didx] is not None:
                    model_residual_func = self._get_model_residual_func(didx)
                    rho = self._param_instance.influence_func_instance.rho
                    for d, w, s in zip(
                            self._data[didx], self._weight[didx], self._scale[didx], strict=True
                    ):
                        residual = model_residual_func(d)
                        tot += w * rho(residual @ residual, s)  # scale

                    self._residual_size[didx] = len(residual)
        else:
            tot = self._evaluator_instance.objective_func(
                model,
                model_ref,
                self._param_instance.influence_func_instance,
                self._data,
                self._weight,
                self._scale)
            self._evaluator_instance.set_residual_size(self._residual_size)

        return tot

    # model_weighted_fit is public to allow result to be checked externally
    def model_weighted_fit(self,
                           *,
                           weight: npt.ArrayLike = None):
        if weight is None:
            weight = self._weight

        if self._data[1] is None:
            return self._model_instance.weighted_fit(
                self._data[0], weight[0], self._scale[0]
            )
        elif self._data[2] is None:
            return self._model_instance.weighted_fit(
                self._data[0],
                weight[0],
                self._scale[0],
                data2=self._data[1],
                weight2=weight[1],
                scale2=self._scale[1],
            )
        else:
            return self._model_instance.weighted_fit(
                self._data[0],
                weight[0],
                self._scale[0],
                data2=self._data[1],
                weight2=weight[1],
                scale2=self._scale[1],
                data3=self._data[2],
                weight3=weight[2],
                scale3=self._scale[2],
            )

    def _init_model(
            self,
            model_start: npt.ArrayLike,
            model_ref_start: npt.ArrayLike) -> None:
        if model_start is None and model_ref_start is None:
            if self._evaluator_instance is not None or callable(self._linear_model_size):
                return self.weighted_fit()
            else:
                return self.model_weighted_fit()
        else:
            return np.copy(model_start), np.copy(model_ref_start)

    def _calc_residual_derivatives(
            self,
            model: npt.ArrayLike,
            *,
            model_ref=None,
            small_diff: float = 1.0e-5
    ) -> (np.ndarray, np.ndarray):
        # build arrays of residuals and gradients per data item
        self._model_instance.cache_model(model, model_ref=model_ref)
        residual_arr = [None] * self._dsize
        residual_gradient_arr = [None] * self._dsize
        for didx in range(self._dsize):
            if self._data[didx] is not None:
                model_residual_func = self._get_model_residual_func(didx)

                residual_arr[didx] = np.zeros(
                    (len(self._data[didx]), self._residual_size[didx])
                )
                residual_gradient_arr[didx] = np.zeros(
                    (len(self._data[didx]), self._residual_size[didx], len(model))
                )

                # first the residuals
                for i, d in enumerate(self._data[didx]):
                    residual_arr[didx][i] = model_residual_func(d)

                # now the gradients
                if self.numeric_derivs_model:
                    model_copy = np.copy(model)
                    for i in range(len(model)):
                        model_copy[i] -= small_diff
                        self._model_instance.cache_model(
                            model_copy, model_ref=model_ref
                        )
                        residual_n_arr = np.zeros(
                            (len(self._data[didx]), self._residual_size[didx])
                        )
                        for j, d in enumerate(self._data[didx]):
                            residual_n_arr[j] = model_residual_func(d)

                        model_copy[i] += 2.0 * small_diff
                        self._model_instance.cache_model(
                            model_copy, model_ref=model_ref
                        )
                        for j, d in enumerate(self._data[didx]):
                            residual = model_residual_func(d)
                            for k in range(self._residual_size[didx]):
                                residual_gradient_arr[didx][(j, k, i)] = (
                                    0.5
                                    * (residual[k] - residual_n_arr[j][k])
                                    / small_diff
                                )

                        model_copy[i] = model[i]
                else:
                    model_residual_gradient_func = (
                        self._get_model_residual_gradient_func(didx)
                    )
                    for j, d in enumerate(self._data[didx]):
                        residual_gradient_arr[didx][j] = model_residual_gradient_func(d)

        return residual_arr, residual_gradient_arr

    def _initialize_residual_size_if_necessary(self) -> None:
        if self._residual_size is None:
            self._residual_size = [None] * self._dsize
            for didx in range(self._dsize):
                if self._data[didx] is not None:
                    residual = self._model_instance.residual(self._data[didx][0])
                    self._residual_size[didx] = len(residual)

    def weighted_fit(
            self,
            weight_arr: list[npt.ArrayLike] = None):
        if self._evaluator_instance is None:
            model = np.zeros(self._linear_model_size())
            self._model_instance.cache_model(model)
            self._initialize_residual_size_if_necessary()

            small_diff = 1.0e-5  # in case numerical differentiation is specified
            residual_arr, residual_gradient_arr = self._calc_residual_derivatives(
                model, small_diff=small_diff
            )

            atot = np.zeros(len(model))
            Atot = np.zeros((len(model), len(model)))
            for didx in range(self._dsize):
                if self._data[didx] is not None:
                    if weight_arr is None or weight_arr[didx] is None:
                        weight = self._weight[didx]
                    else:
                        weight = weight_arr[didx]

                    residual = residual_arr[didx]
                    residual_gradient = residual_gradient_arr[didx]
                    for i, (w, s) in enumerate(zip(weight, self._scale[didx], strict=True)):
                        grad = np.matmul(np.transpose(residual_gradient[i]), residual[i])
                        w /= s * s
                        atot += w * grad
                        Atot += w * np.matmul(
                            np.transpose(residual_gradient[i]), residual_gradient[i]
                        )

            if self._messages_file is not None:
                print("Here Atot",Atot, file=self._messages_file)
                print("Here atot",atot, file=self._messages_file)

            return -np.linalg.solve(Atot, atot),None
        else:
            return self._evaluator_instance.weighted_fit(self._data,
                                                         self._weight if weight_arr is None else weight_arr,
                                                         self._scale)

    def __residual_influence(
        self,
        residual: np.ndarray,
        scale: npt.ArrayLike,
        small_diff: float = 1.0e-5
    ) -> float:
        rsqr = residual @ residual
        if self.numeric_derivs_influence:
            r = math.sqrt(rsqr)
            rho_n = self._param_instance.influence_func_instance.rho(
                (r - small_diff) ** 2.0, scale
            )
            rho_p = self._param_instance.influence_func_instance.rho(
                (r + small_diff) ** 2.0, scale
            )
            rho_deriv = 0.5 * (rho_p - rho_n) / small_diff
            return rho_deriv / r
        else:
            return self._param_instance.influence_func_instance.rhop(rsqr, scale)

    # update_weights is public to allow result to be checked externally
    def update_weights(
        self,
        model: npt.ArrayLike,
        weight: npt.ArrayLike,
        *,
        model_ref=None
    ) -> None:
        if self._evaluator_instance is None:
            self._model_instance.cache_model(model, model_ref=model_ref)
            obj_func_sign = (
                self._param_instance.influence_func_instance.objective_func_sign()
            )
            for didx in range(self._dsize):
                if self._data[didx] is not None:
                    model_residual_func = self._get_model_residual_func(didx)
                    for i, (d, s) in enumerate(
                            zip(self._data[didx], self._scale[didx], strict=True)
                    ):
                        weight[didx][i] = (
                            self._weight[didx][i]
                            * obj_func_sign
                            * self.__residual_influence(model_residual_func(d), s)
                        )
        else:
            self._evaluator_instance.update_weights(
                model,
                model_ref,
                self._param_instance.influence_func_instance,
                self._data,
                self._weight,
                self._scale,
                weight
            )
            
    def finalise(self,
                 model,
                 *,
                 model_ref=None,
                 weight=None,
                 itn:int=0,
                 total_time=0):
        self._param_instance.reset(init=False)

        self.final_model = model
        self.final_model_ref = model_ref
        if weight is None:
            weight = [None] * self._dsize
            for didx in range(self._dsize):
                if self._data[didx] is not None:
                    weight[didx] = np.copy(self._weight[didx])

        self.update_weights(model, weight, model_ref=model_ref)
        weight_scale = self._param_instance.influence_func_instance.objective_func_sign()*self._param_instance.influence_func_instance.rhop(0.0,1.0)

        if weight[0] is None:
            self.final_weight = None
        else:
            self.final_weight = weight[0]/weight_scale

        if weight[1] is None:
            self.final_weight2 = None
        else:
            self.final_weight2 = weight[1]/weight_scale

        if weight[2] is None:
            self.final_weight3 = None
        else:
            self.final_weight3 = weight[2]/weight_scale

        if self._debug:
            self.debug_n_iterations = itn + 1
            self.debug_total_time = total_time

