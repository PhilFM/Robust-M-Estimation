import math
import numpy as np
import numpy.typing as npt
from typing import TextIO
import time

try:
    from .base_irls import BaseIRLS
except ImportError:
    from base_irls import BaseIRLS

class SupGaussNewton(BaseIRLS):
    def __init__(
        self,
        param_instance,
        data: npt.ArrayLike,
        *,
        model_instance = None, # Python model
        evaluator_instance = None, # Cython model
        weight: npt.ArrayLike = None,
        scale: npt.ArrayLike = None,
        data2: npt.ArrayLike = None,
        weight2: npt.ArrayLike = None,
        scale2: npt.ArrayLike = None,
        data3: npt.ArrayLike = None,
        weight3: npt.ArrayLike = None,
        scale3: npt.ArrayLike = None,
        numeric_derivs_model: bool = False,
        numeric_derivs_influence: bool = False,
        max_niterations: int = 50,
        residual_tolerance: float = 1.0e-8,
        lambda_start: float = 1.0,
        lambda_max: float = 1.0,
        lambda_scale: float = 1.2,
        lambda_thres: float = 0.0,
        diff_thres: float = 1.0e-10,
        model_size_est: npt.ArrayLike = None,
        messages_file: TextIO = None,
        debug: bool = False,
    ):
        BaseIRLS.__init__(
            self,
            param_instance,
            data,
            model_instance=model_instance,
            evaluator_instance=evaluator_instance,
            weight=weight,
            scale=scale,
            data2=data2,
            weight2=weight2,
            scale2=scale2,
            data3=data3,
            weight3=weight3,
            scale3=scale3,
            numeric_derivs_model=numeric_derivs_model,
            numeric_derivs_influence=numeric_derivs_influence,
            max_niterations=max_niterations,
            diff_thres=diff_thres,
            messages_file=messages_file,
            debug=debug,
        )
        self.__residual_tolerance = residual_tolerance
        self.__lambda_start = lambda_start
        self.__lambda_max = lambda_max
        self.__lambda_scale = lambda_scale
        self.__lambda_thres = lambda_thres
        self.__model_size_est = model_size_est

    def __calc_influence_func_derivatives(
        self, residual: np.ndarray,
        s: float,
        *,
        small_diff: float = 1.0e-5
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
            rho_c = self._param_instance.influence_func_instance.rho(r * r, s)  # scale
            rho_2nd_deriv = (rho_n + rho_p - 2.0 * rho_c) / (small_diff * small_diff)
            Bterm = (r * rho_2nd_deriv - rho_deriv) / (r * r * r)
        else:
            rhop = self._param_instance.influence_func_instance.rhop(rsqr, s)  # scale
            Bterm = self._param_instance.influence_func_instance.Bterm(
                rsqr, s
            )  # scale

        return rhop, Bterm

    # additional calculation of derivatives for calculating GNC schedule
    def __calc_influence_func_derivatives_iv(
        self, residual: np.ndarray,
        s: float,
        *,
        small_diff: float = 1.0e-5
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
            rho_c = self._param_instance.influence_func_instance.rho(r * r, s)  # scale
            rho_2nd_deriv = (rho_n + rho_p - 2.0 * rho_c) / (small_diff * small_diff)
            Bterm = (r * rho_2nd_deriv - rho_deriv) / (r * r * r)

            # 2nd derivative w.r.t. x and inv_var
            rho_nn = self._param_instance.influence_func_instance.rho(
                (r - small_diff) ** 2.0, s*(1. - small_diff)
            )
            rho_pn = self._param_instance.influence_func_instance.rho(
                (r + small_diff) ** 2.0, s*(1. - small_diff)
            )
            rho_np = self._param_instance.influence_func_instance.rho(
                (r - small_diff) ** 2.0, s*(1. + small_diff)
            )
            rho_pp = self._param_instance.influence_func_instance.rho(
                (r + small_diff) ** 2.0, s*(1. + small_diff)
            )
            rho_deriv_piv = 0.25*(rho_nn - rho_pn - rho_np + rho_pp) / (small_diff * small_diff)
            rhopiv = rho_deriv_piv / r
        else:
            rhop = self._param_instance.influence_func_instance.rhop(rsqr, s)  # scale
            Bterm = self._param_instance.influence_func_instance.Bterm(
                rsqr, s
            )  # scale
            rhopiv = self._param_instance.influence_func_instance.rhopiv(rsqr, s) # scale

        return rhop, Bterm, rhopiv

    # weighted_derivs is public to allow derivatives to be checked
    def weighted_derivs(
            self,
            model: npt.ArrayLike,
            lambda_b: float,
            *,
            model_ref=None
    ) -> (np.array, np.array):
        if self._evaluator_instance is None:
            # initialize residual_size
            self._model_instance.cache_model(model, model_ref=model_ref)
            self._initialize_residual_size_if_necessary()

            small_diff = 1.0e-5
            residual_arr, residual_gradient_arr = self._calc_residual_derivatives(
                model, model_ref=model_ref, small_diff=small_diff
            )

            atot = np.zeros(len(model))
            AlBtot = np.zeros((len(model), len(model)))
            for didx in range(self._dsize):
                if self._data[didx] is not None:
                    residual = residual_arr[didx]
                    residual_gradient = residual_gradient_arr[didx]
                    for i, (w, s) in enumerate(
                            zip(self._weight[didx], self._scale[didx], strict=True)
                    ):
                        grad = np.matmul(np.transpose(residual_gradient[i]), residual[i])
                        rhop, Bterm = self.__calc_influence_func_derivatives(
                            residual[i], s, small_diff=small_diff
                        )

                        atot += w * rhop * grad
                        AlBtot += w * (
                            rhop
                            * np.matmul(
                                np.transpose(residual_gradient[i]), residual_gradient[i]
                            )
                            + lambda_b * Bterm * np.outer(grad, grad)
                        )

            return atot, AlBtot
        else:
            return self._evaluator_instance.weighted_derivs(
                model,
                model_ref,
                self._param_instance.influence_func_instance,
                lambda_b,
                self._data,
                self._weight,
                self._scale)

    # derivatives representing size of distribution
    def weighted_gnc_derivs(
            self,
            model: npt.ArrayLike,
            *,
            model_ref=None
    ) -> (np.array, np.array):
        if self._evaluator_instance is None:
            # initialize residual_size
            self._model_instance.cache_model(model, model_ref=model_ref)
            self._initialize_residual_size_if_necessary()

            small_diff = 1.0e-5
            residual_arr, residual_gradient_arr = self._calc_residual_derivatives(
                model, model_ref=model_ref, small_diff=small_diff
            )

            aivtot = np.zeros(len(model))
            Aivtot = np.zeros((len(model), len(model)))
            for didx in range(self._dsize):
                if self._data[didx] is not None:
                    residual = residual_arr[didx]
                    residual_gradient = residual_gradient_arr[didx]
                    for i, (w, s) in enumerate(
                            zip(self._weight[didx], self._scale[didx], strict=True)
                    ):
                        grad = np.matmul(np.transpose(residual_gradient[i]), residual[i])
                        rhop, Bterm, rhopiv = self.__calc_influence_func_derivatives_iv(
                            residual[i], s, small_diff=small_diff
                        )

                        aivtot += w * rhopiv * grad
                        Aivtot += w * (
                            rhop
                            * np.matmul(
                                np.transpose(residual_gradient[i]), residual_gradient[i]
                            )
                            + Bterm * np.outer(grad, grad)
                        )

            return aivtot, Aivtot
        else:
            return self._evaluator_instance.weighted_gnc_derivs(
                model,
                model_ref,
                self._param_instance.influence_func_instance,
                self._data,
                self._weight,
                self._scale)

    # estimate of change in inverse variance representing size of distribution
    def gnc_normalised_deriv(  self,
                        model_size_est: npt.ArrayLike,
                        model: npt.ArrayLike,
                        *,
                        model_ref=None
    ) -> np.ndarray:
        aivtot, Aivtot = self.weighted_gnc_derivs(model,
                                                  model_ref=model_ref)
        v = np.linalg.solve(Aivtot, aivtot)
        vp = v / model_size_est
        vpp = vp / math.sqrt(self._param_instance.influence_func_instance.variance())
        if self._messages_file is not None:
            print("size=",math.sqrt(self._param_instance.influence_func_instance.variance()),"vp=",vp,"vpp=",vpp,"model=",model)

        return vpp

    def run(self,
            *,
            model_start: npt.ArrayLike = None,
            model_ref_start: npt.ArrayLike=None) -> bool:
        self._param_instance.reset()
        lambda_val = self.__lambda_start
        model, model_ref = self._init_model(model_start, model_ref_start)
        last_tot = self.objective_func(model, model_ref=model_ref)

        model_is_valid = getattr(self._model_instance, "model_is_valid", None)
        update_model_ref = getattr(self._model_instance, "update_model_ref", None)

        if self._messages_file is not None:
            a, AlB = self.weighted_derivs(model, self.__lambda_start, model_ref=model_ref)
            print("Initial model=", model, file=self._messages_file)
            print("Initial model_ref=", model_ref, file=self._messages_file)
            print(
                "Initial params=",
                self._param_instance.influence_func_instance.summary(),
                file=self._messages_file
            )
            print(
                "Initial tot=",
                last_tot,
                "grad=",
                a.dtype, a,
                "diff_thres=",
                self._diff_thres,
                file=self._messages_file
            )
            print("Initial weighted derivative lambda_val=", lambda_val, ":", file=self._messages_file)
            print("Initial AlB=", AlB.dtype, AlB, file=self._messages_file)

        if self._debug:
            self.debug_diffs = []
            self.debug_diff_alpha = []
            self.debug_model_list = []
            self.debug_model_list.append(
                (
                    0.0,  # iteration alpha
                    np.copy(model),
                    self._param_instance.alpha(), # GNC alpha
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
            model_ref_old = model_ref

            if self.__lambda_thres == 0.0 or lambda_val > self.__lambda_thres:
                lambda_a = 1.0
            else:
                lambda_a = lambda_val/self.__lambda_thres

            lambda_b = max(0.0, lambda_val - self.__lambda_thres)
            a, AlB = self.weighted_derivs(model_old, lambda_b, model_ref=model_ref_old)

            if self._debug:
                self.debug_weighted_derivs_time += time.time() - start_time
                start_time = time.time()

            try:
                at = np.linalg.solve(AlB, a)
            except np.linalg.LinAlgError:
                lambda_val /= self.__lambda_scale
                if self._messages_file is not None:
                    print(
                        "Weighted derivative matrix is singular - rejecting new lambda_val=",
                        lambda_val,
                        file=self._messages_file
                    )

                continue

            gnc_alpha = self._param_instance.alpha()
            gnc_filter_size = self._param_instance.filter_size()
            if self._debug:
                self.debug_solve_time += time.time() - start_time

            at *= lambda_a
            model -= at
            if callable(update_model_ref):
                model_ref = update_model_ref(model, model_ref)

            # check whether the model is valid, abort this iteration if it isn't
            if callable(model_is_valid):
                if not model_is_valid(model, model_ref):
                    model = model_old
                    model_ref = model_ref_old
                    lambda_val /= self.__lambda_scale
                    if self._messages_file is not None:
                        print(
                            "Aborting here new lambda=",
                            lambda_val,
                            file=self._messages_file)

                    continue

            tot = self.objective_func(model, model_ref=model_ref)
            tot_diff = self._param_instance.influence_func_instance.objective_func_sign() * (tot - last_tot)

            # only check for termination if we have reached the end of any GNC schedule
            if self._diff_thres is not None and gnc_alpha == 1.0:
                model_max_diff = np.linalg.norm(at, ord=np.inf)
                if self._debug is True and model_max_diff > 0.0:
                    self.debug_diffs.append(math.log10(model_max_diff))
                    self.debug_diff_alpha.append(gnc_alpha)

                if tot_diff <= 0.0 and model_max_diff < self._diff_thres:
                    if self._messages_file is not None:
                        print("Difference threshold reached", file=self._messages_file)

                    all_good = True
                    break

            if self._messages_file:
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
                    file=self._messages_file
                )

            if (
                lambda_val != 0.0
                and tot_diff > self.__residual_tolerance
            ):
                if self._messages_file is not None:
                    print(
                        "Reject lambda_val=",
                        lambda_val,
                        "diff=",
                        tot_diff,
                        "reverting to model",
                        model_old,
                        file=self._messages_file
                    )

                model = model_old
                model_ref = model_ref_old
                lambda_val /= self.__lambda_scale
                last_tot = self.objective_func(model, model_ref=model_ref)
            else:
                if self._messages_file is not None:
                    print("Accept lambda_val=", lambda_val, "model=", model, file=self._messages_file)

                lambda_val = min(self.__lambda_scale * lambda_val, self.__lambda_max)

                if self._param_instance.supports_factor_argument() and self.__model_size_est is not None:
                    normalised_deriv = self.gnc_normalised_deriv(self.__model_size_est, model, model_ref=model_ref)
                    abs_normalised_deriv = np.abs(normalised_deriv)
                    self._param_instance.increment(max(abs_normalised_deriv))
                else:
                    self._param_instance.increment()

                if self._param_instance.alpha() == 1.0:
                    last_tot = tot
                else:
                    last_tot = self.objective_func(model, model_ref=model_ref)

            if self._debug:
                self.debug_model_list.append(
                    (
                        (1 + itn) / (self._max_niterations - 1),  # alpha
                        np.copy(model),
                        gnc_filter_size,
                    )
                )

        self.finalise(model, model_ref=model_ref, itn=itn, total_time = time.time() - start_time_total if self._debug else 0)
        return all_good

    def finalise(self,
                 model,
                 *,
                 model_ref=None,
                 weight=None,
                 itn:int=0,
                 total_time=0):
        a, AlB = self.weighted_derivs(model, 1.0, model_ref=model_ref)
        BaseIRLS.finalise(
            self,
            model,
            model_ref=model_ref,
            AlB=AlB,
            weight=weight,
            itn=itn,
            total_time=total_time,
            )
        
    
