import math
import numpy as np
import time

from .base_irls import BaseIRLS


class IRLS(BaseIRLS):
    def __init__(
        self,
        param_instance,
        model_instance,
        data,
        data_ids=None,
        weight=None,
        scale=None,
        numeric_derivs_influence: bool = False,
        max_niterations: int = 50,
        diff_thres:float = 1.0e-12,
        print_warnings:bool = False,
        model_start=None,
        model_ref_start=None,
        debug:bool = False,
    ):
        BaseIRLS.__init__(
            self,
            param_instance,
            model_instance,
            data,
            data_ids=data_ids,
            weight=weight,
            scale=scale,
            numeric_derivs_influence=numeric_derivs_influence,
            max_niterations=max_niterations,
            diff_thres=diff_thres,
            print_warnings=print_warnings,
            model_start=model_start,
            model_ref_start=model_ref_start,
            debug=debug,
        )

    def __updated_weight(
        self, residual: np.array, scale, small_diff: float = None
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

    def __update_weights(self, model, weight, model_ref=None) -> None:
        small_diff = 1.0e-5  # in case numerical differentiation is specified
        self._model_instance.cache_model(model, model_ref=model_ref)
        for i, (d, di, s) in enumerate(
                zip(self._data, self._data_ids, self._scale, strict=True)
            ):
                weight[i] = self._weight[i] * self.__updated_weight(
                    self._model_instance.residual(d, di), s
                )

    def run(self) -> bool:
        self._param_instance.reset()
        weight = np.copy(self._weight)
        model, model_ref = self._init_model()
        if self._print_warnings:
            print(
                "Initial model=",
                model,
                "params=",
                self._param_instance.influence_func_instance.summary(),
                "diff_thres=",
                self._diff_thres,
            )

        if self._debug:
            self.debug_diffs = []
            self.debug_diff_alpha = []
            self.debug_model_list = []
            self.debug_model_list.append(
                (
                    0.0,  # alpha
                    np.copy(model),
                )
            )

            self.debug_update_weights_time = 0.0
            self.debug_weighted_fit_time = 0.0
            self.debug_total_time = 0.0
            start_time_total = time.time()

        all_good = False
        for itn in range(self._max_niterations):
            if self._debug:
                start_time = time.time()

            self.__update_weights(model, weight, model_ref=model_ref)
            if self._debug:
                self.debug_update_weights_time += time.time()-start_time
                start_time = time.time()

            model_old = model
            if callable(self._linear_model_size):
                model = self.weighted_fit(weight)
            else:
                model, model_ref = self._model_instance.weighted_fit(
                    self._data, self._data_ids, weight, self._scale
                )

            if self._debug:
                self.debug_weighted_fit_time += time.time()-start_time

            if self._param_instance.alpha() == 1.0:
                if self._diff_thres is not None:
                    model_max_diff = np.linalg.norm(model - model_old, ord=np.inf)
                    if self._print_warnings:
                        print("model_max_diff=", model_max_diff)

                    if self._debug is True and model_max_diff > 0.0:
                        if self._print_warnings:
                            print("Adding diff model_max_diff", model_max_diff)

                        self.debug_diffs.append(math.log10(model_max_diff))
                        self.debug_diff_alpha.append(self._param_instance.alpha())

                    if model_max_diff < self._diff_thres:
                        if self._print_warnings:
                            print("Difference threshold reached")

                        all_good = True
                        break

            if self._print_warnings:
                print(
                    "itn=",
                    itn,
                    "model=",
                    model,
                    "params=",
                    self._param_instance.influence_func_instance.summary(),
                )

            self._param_instance.increment()
            if self._debug:
                self.debug_model_list.append(
                    (
                        (1+itn) / (self._max_niterations - 1),  # alpha
                        np.copy(model),
                    )
                )

        self._param_instance.reset(
            False
        )

        # finish with parameters in correct final model
        self.final_model = model
        self.final_model_ref = model_ref

        if self._debug:
            self.debug_n_iterations = itn + 1
            self.debug_total_time = time.time()-start_time_total

        return all_good
