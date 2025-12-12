import math
import numpy as np
import time

from .base_irls import BaseIRLS


class SupGaussNewton(BaseIRLS):
    def __init__(
        self,
        param_instance,
        model_instance,
        data,
        data_ids=None,
        weight=None,
        scale=None,
        numeric_derivs_model: bool = False,
        numeric_derivs_influence: bool =False,
        max_niterations: int = 50,
        residual_tolerance: float = 1.0e-8,
        lambda_start: float = 1.0,
        lambda_max: float = 1.0,
        lambda_scale: float = 1.2,
        diff_thres: float = 1.0e-10,
        print_warnings: bool = False,
        model_start = None,
        model_ref_start = None,
        model_range = None,
        debug: bool = False,
    ):
        BaseIRLS.__init__(self,
            param_instance,
            model_instance,
            data,
            data_ids=data_ids,
            weight=weight,
            scale=scale,
            numeric_derivs_model=numeric_derivs_model,
            numeric_derivs_influence=numeric_derivs_influence,
            max_niterations=max_niterations,
            diff_thres=diff_thres,
            print_warnings=print_warnings,
            model_start=model_start,
            model_ref_start=model_ref_start,
            debug=debug,
        )
        self.__residual_tolerance = residual_tolerance
        self.__lambda_start = lambda_start
        self.__lambda_max = lambda_max
        self.__lambda_scale = lambda_scale
        self.__model_range = model_range

    def __calc_influence_func_derivatives(
        self, residual: np.array, s: float, small_diff: float = 1.0e-5
    ) -> (float, float):  # scale
        rsqr = residual @ residual
        r = math.sqrt(rsqr)
        if self.numeric_derivs_influence:
            rho_n = self._param_instance.influence_func_instance.rho(
                (r - small_diff) ** 2.0, s
            )  # scale
            rho_p = self._param_instance.influence_func_instance.rho(
                (r + small_diff) ** 2.0, s
            )  # scale
            rho_deriv = 0.5 * (rho_p - rho_n) / small_diff
            rhop = rho_deriv / r
            rho_c = self._param_instance.influence_func_instance.rho(
                r * r, s
            )  # scale
            rho_2nd_deriv = (rho_n + rho_p - 2.0 * rho_c) / (small_diff * small_diff)
            Bterm = (r * rho_2nd_deriv - rho_deriv) / (r * r * r)
        else:
            rhop = self._param_instance.influence_func_instance.rhop(
                rsqr, 1.0
            )  # scale
            Bterm = self._param_instance.influence_func_instance.Bterm(
                rsqr, 1.0
            )  # scale

        return rhop, Bterm

    # weighted_derivs is public to allow derivatives 
    def weighted_derivs(
        self, model, lambda_val: float, weight=None, model_ref=None
    ) -> (np.array, np.array):
        if weight is None:
            weight = self._weight

        # initialize residual_size
        self._model_instance.cache_model(model, model_ref = model_ref)
        residual = self._model_instance.residual(self._data[0], self._data_ids[0])
        self._residual_size = len(residual)

        small_diff = 1.0e-5
        residual_arr, residual_gradient_arr = self._calc_residual_derivatives(model, model_ref = model_ref, small_diff = small_diff)

        atot = np.zeros(len(model))
        AlBtot = np.zeros((len(model), len(model)))
        for i,(w,s) in enumerate(zip(weight, self._scale, strict=True)):
            grad = np.matmul(np.transpose(residual_gradient_arr[i]), residual_arr[i])
            rhop, Bterm = self.__calc_influence_func_derivatives(
                residual_arr[i], s, small_diff=small_diff
            )
            atot += w * rhop * grad
            AlBtot += w * (
                rhop * np.matmul(np.transpose(residual_gradient_arr[i]), residual_gradient_arr[i])
                + lambda_val * Bterm * np.outer(grad, grad)
            )

        return atot, AlBtot

    def run(self) -> bool:
        self._param_instance.reset()
        lambda_val = self.__lambda_start
        model, model_ref = self._init_model()
        last_tot = self.objective_func(model, model_ref=model_ref)
        update_model_ref = getattr(self._model_instance, "update_model_ref", None)

        if self._print_warnings:
            a, AlB = self.weighted_derivs(model, lambda_val, model_ref=model_ref)
            print("Initial model=", model)
            print("Initial model_ref=", model_ref)
            print(
                "Initial params=",
                self._param_instance.influence_func_instance.summary(),
            )
            print(
                "Initial tot=",
                last_tot,
                "grad=",
                a,
                "diff_thres=",
                self._diff_thres,
            )
            print("Initial weighted derivative lambda_val=", lambda_val, ":")
            print("Initial AlB=", AlB)

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

            self.debug_weighted_derivs_time = 0.0
            self.debug_solve_time = 0.0
            start_time_total = time.time()

        all_good = False
        for itn in range(self._max_niterations):
            if self._debug:
                start_time = time.time()

            model_old = model.copy()
            model_refOld = model_ref
            a, AlB = self.weighted_derivs(model_old, lambda_val, model_ref=model_ref)
            if self._debug:
                self.debug_weighted_derivs_time += time.time()-start_time
                start_time = time.time()

            try:
                at = np.linalg.solve(AlB, a)
            except np.linalg.LinAlgError:
                lambda_val /= self.__lambda_scale
                if self._print_warnings:
                    print(
                        "Weighted derivative matrix is singular - rejecting new lambda_val=",
                        lambda_val,
                    )

                continue

            if self._debug:
                self.debug_solve_time += time.time()-start_time

            model -= at
            if callable(update_model_ref):
                model_ref = update_model_ref(model, model_ref)

            # compare against any provided range
            if self.__model_range is not None:
                all_good_range = True
                for i in range(len(model)):
                    this_range = self.__model_range[i]
                    if this_range is not None:
                        if model[i] < this_range[0] or model[i] > this_range[1]:
                            # outside range - revert
                            if self._print_warnings:
                                print(
                                    "Reject lambda_val=",
                                    lambda_val,
                                    "outside range ",
                                    model[i],
                                    " at model parameter ",
                                    i,
                                    ": reverting to model",
                                    model_old
                                )

                            model = model_old
                            model_ref = model_refOld
                            lambda_val /= self.__lambda_scale
                            all_good_range = False

                if not all_good_range:
                    continue

            if self._diff_thres is not None:
                model_max_diff = np.linalg.norm(at, ord=np.inf)
                if self._debug is True and model_max_diff > 0.0:
                    self.debug_diffs.append(math.log10(model_max_diff))
                    self.debug_diff_alpha.append(self._param_instance.alpha())

                if model_max_diff < self._diff_thres:
                    if self._print_warnings:
                        print("Difference threshold reached")

                    all_good = True
                    break

            tot = self.objective_func(model, model_ref=model_ref)
            if self._print_warnings:
                print(
                    "itn=",
                    itn,
                    "model=",
                    model,
                    "model_old=",
                    model_old,
                    "params=",
                    self._param_instance.influence_func_instance.summary(),
                    "tot=",
                    tot,
                )

            if (
                lambda_val != 0.0
                and self._param_instance.influence_func_instance.objective_func_sign()
                * (tot - last_tot)
                > self.__residual_tolerance
            ):
                if self._print_warnings:
                    print(
                        "Reject lambda_val=",
                        lambda_val,
                        "diff=",
                        last_tot - tot,
                        "reverting to model",
                        model_old,
                    )

                model = model_old
                model_ref = model_refOld
                lambda_val /= self.__lambda_scale
                last_tot = self.objective_func(model, model_ref=model_ref)
            else:
                if self._print_warnings:
                    print("Accept lambda_val=", lambda_val, "model=", model)

                lambda_val = min(self.__lambda_scale * lambda_val, self.__lambda_max)
                self._param_instance.increment()
                if self._param_instance.alpha() == 1.0:
                    last_tot = tot
                else:
                    last_tot = self.objective_func(model, model_ref=model_ref)

            if self._debug:
                self.debug_model_list.append(
                    (
                        (1+itn) / (self._max_niterations - 1),  # alpha
                        np.copy(model),
                    )
                )

        self._param_instance.reset(
            False
        )  # finish with parameters in correct final model

        self.final_model = model
        self.final_model_ref = model_ref

        if self._debug:
            self.debug_n_iterations = itn + 1
            self.debug_total_time = time.time()-start_time_total

        return all_good
