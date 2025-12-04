import math
import numpy as np

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
        model_start=None,
        model_ref_start=None,
        debug: bool = False,
    ):
        BaseIRLS.__init__(self,
            param_instance,
            model_instance,
            data,
            data_ids,
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

    def run(self):
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
            diffs = []
            model_list = []

        all_good = True
        for itn in range(self._max_niterations):
            model_old = model.copy()
            model_refOld = model_ref
            a, AlB = self.weighted_derivs(model_old, lambda_val, model_ref=model_ref)
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

            model -= at
            if callable(update_model_ref):
                model_ref = update_model_ref(model, model_ref)

            if (
                self._param_instance.at_final_stage()
                and self._diff_thres is not None
            ):
                model_max_diff = np.linalg.norm(at, ord=np.inf)
                if self._debug is True and model_max_diff > 0.0:
                    diffs.append(math.log10(model_max_diff))

                if model_max_diff < self._diff_thres:
                    if self._print_warnings:
                        print("Difference threshold reached")

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
                model = model_old
                model_ref = model_refOld
                lambda_val /= self.__lambda_scale
                last_tot = self.objective_func(model, model_ref=model_ref)
                if self._print_warnings:
                    print(
                        "Reject new lambda_val=",
                        lambda_val,
                        "diff=",
                        last_tot - tot,
                        "reverting to model",
                        model,
                    )
            else:
                lambda_val = min(self.__lambda_scale * lambda_val, self.__lambda_max)
                self._param_instance.update()
                if self._param_instance.at_final_stage():
                    last_tot = tot
                else:
                    last_tot = self.objective_func(model, model_ref=model_ref)

                if self._print_warnings:
                    if lambda_val != 0.0 and self.__lambda_scale > 1.0:
                        print("Accept new lambda_val=", lambda_val)

            if self._debug:
                model_list.append(
                    (
                        itn / (self._max_niterations - 1),  # alpha
                        np.copy(model),
                    )
                )

        self._param_instance.reset(
            False
        )  # finish with parameters in correct final model
        if self._debug:
            if model_ref is None:
                if all_good is True:
                    return model, itn + 1, diffs, model_list
                else:
                    return None, None, None, None
            else:
                if all_good is True:
                    return model, model_ref, itn + 1, diffs, model_list
                else:
                    return None, None, None, None, None
        else:
            if model_ref is None:
                if all_good is True:
                    return model
                else:
                    return None
            else:
                if all_good is True:
                    return model, model_ref
                else:
                    return None, None
