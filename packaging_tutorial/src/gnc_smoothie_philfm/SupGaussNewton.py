# Supervised Gauss-Newton (Sup-GN) algorithm, an alternative to IRLS most suitable
# for the two cases:
#   * Linear model where the data relates to the model via a linear function.
#   * Non-linear model where there is no closed-form solution to calculating the
#     model parameters from the weighted data.
# Use the basic IRLS in the remaining case where a non-trivial closed-form solution for the model
# is available, such as 3D point cloud registration (SVD solution). IRLS is not suitable for
# non-linear problems where a closed-form solution is not available, but Sup-GN can be used in
# such problems so long as a reasonable starting point for the model can be supplied (see the
# model_start and model_ref_start parameters below). For linear models Sup-GN provides a simpler
# model implementation than IRLS, since the closed-form solution for the model is calculated
# internally. Also Sup-GN converges quadratically for linear models when close to the solution.
#
# Parameters:
#   param_instance: Defines the GNC schedule to be followed by IRLS. If GNC is not being used then
#                   this can be a NullParams instance. Should have an internal influence_func_instance
#                   that specifies the IRLS influence function to be used. This influence_func_instance
#                   should provide these methods:
#                   * objective_func_sign(self) -> float:
#                     Returns either one or minus one depending on whether the objective function
#                     increases for large residuals (one) or decreases (minus one). Typical IRLS
#                     objective functions such as Huber and Geman-McClure increase for large residuals,
#                     so most functions will return one. The version of Welsch we have implemented in
#                     GNC_WelschParams.py uses a nagative sense, which slightly simplifies the
#                     implementation, because otherwise we would have to add one to the objective
#                     function in order to keep it positive.
#                   * rho(self, rsqr: float, s: float) -> float:
#                     The objective function given
#                     - rsqr: The square of the L2 norm of the residual vector
#                     - s: The scale of the data item indicating its known inaccuracy, so a value >= 1.
#                     Returns the value of the objective function.
#                   * rhop(self, rsqr: float, s: float) -> float:
#                     The influence function, which is equal to the derivative with respect to r
#                     of rho(rsqr,s) divided by r, where r is the L2 norm of the residual vector.
#                     If numeric_derivs_influence is set to True (see below) then the derivatives
#                     are calculated numerically from rho() and rhop() is not required.
#                   * Bterm(self, rsqr: float, s: float) -> float:
#                     Implements (r*rho''(r) - rho'(r))/(r^3) where ' indicates derivative.
#                     If numeric_derivs_influence is set to True (see below) then the derivatives
#                     are calculated numerically from rho() and Bterm() is not required.
#                   * summary(self) -> str:
#                     A string containing the values of the internal parameters.
#
#                   param_instance itself shoule provide the following methods:
#                   * reset(self, init: bool = True):
#                         Resets the internal influence_func_instance according to the stage of the
#                         GNC schedule indicated by the init parameter. If init is True, reset to the
#                         starting value to prepare for the GNC process to start. If init is False,
#                         reset to the final stage of GNC.
#                   * at_final_state(self) -> bool:
#                         Returns True if the GNC schedule has reached the final stage, False otherwise
#                   * update(self): Update the influence_func_instance to the next step in the GNC schedule.
#   model_instance: The model being fitted to the data, a class instance that provides the following
#                   functions:
#                   * residual(self, model, data_item, model_ref=None) -> np.array:
#                          Calculates the residual (error)
#                          of the data_item given the model. If the model contains reference parameters
#                          e.g. for estimating rotation, these are passed as model_ref.
#                   * residual_gradient(self, model, data_item, model_ref=None) -> np.array:
#                          The Jacobian or derivative matrix of the residual vector with respect
#                          to the model parameters. If the numeric_derivs_model parameter is set to True
#                          (see below) then the derivatives are calculated numerically using the residual()
#                          function.
#                   * linear_model_size(self) -> int:
#                          Returns the number of parameters in the model if the
#                          model is linear, otherwise 0. In the linear case the IRLS class uses an
#                          internal weighted_fit() function to fit the model to the data with specified
#                          weights, so that the programmer does not have to implement it.
#                   * weighted_fit(self, data, weight, scale=None) -> (np.array, np.array):
#                          If linear_model_size() returns 0, the model is not linear. If a closed-form
#                          solution for the best model given the data with weights nevertheless exists,
#                          implement it in the class. The scale
#                          array indicates that certain data items are less accurate and so have a
#                          scale value > 1, indicating that the influence function for that data item
#                          should be stretched by the given scale factor.
#                          For non-linear problems with no closed-form solution, pass a suitable starting
#                          point as the model_start (and optionally model_ref_start) parameters, see below.
#                          In that case the weighted_fit() method is not used.
#  data: An array of data items. Each data item is itself an array.
#  weight: An optional array of float weight values for each data item.
#          If not provided, weights are initialised to one
#  scale: An optional array of scale values, indicating that one or more data items are known to
#         have reduced accuracy, i.e. a wider influence function. The scale indicates the stretching
#         to apply to the influence function for that data item.
#  numeric_derivs_model: Whether to calculate derivatives of the data residual vector with respect to the
#                        model parameters numerically using a provided residual() function or directly
#                        using a provided residual_gradient() function.
#  numeric_derivs_influence: Whether to calculate derivatives of the influence function numerically
#                            from a provided rho() function or directly using a provided rhop() function.
#  max_niterations: Maximum number of Sup-GN iterations to apply before aborting
#  residual_tolerance: An optional parameter that is used to terminate Sup-GN when the improvement to the
#                      objective function value is smaller than the provided threshold
#  lambda_start: Starting value for the Sup-GN damping, similar to Levenberg-Marquart damping.
#                In Sup-GN the level of damping is high when lambda is small, so normally it is
#                best to start with an optimistic small value.
#  lambda_max: Maximum value for lambda in Sup-GN damping. This should be in the range [0,1].
#  lambda_scale: Scale factor to multiply lambda by when an iteration successfully reduces/increases
#                the objective function (depending on the +/- sign specified by
#                param_instance.influence_func_instance.influence_func_sign(), see above).
#                When the iteration is not successful, the model change is reverted and lambda is divided
#                by this factor to increase the damping at the next iteration.
#  diff_thres: Terminate when successful update changes the model parameters by less than this value.
#  print_warnings: Whether to print debugging information.
#  model_start: Optional starting value for model parameters
#  model_ref_start: Optional starting reference parameters for model, e.g. if optimising rotation
#  debug: Whether to return extra debugging data on exit:
#             * The number of iterations actually applied
#             * The norm of the model parameters change at each iteration, as a list of difference values
#             * A list of the model parameters at each iteration
import math
import numpy as np

from .BaseIRLS import BaseIRLS


class SupGaussNewton:
    def __init__(
        self,
        param_instance,
        model_instance,
        data,
        weight=None,
        scale=None,
        numeric_derivs_model: bool = False,
        numeric_derivs_influence=False,
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
        self.base = BaseIRLS(
            param_instance,
            model_instance,
            data,
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
        self.residual_tolerance = residual_tolerance
        self.lambda_start = lambda_start
        self.lambda_max = lambda_max
        self.lambda_scale = lambda_scale

    def calc_influence_func_derivatives(
        self, residual: np.array, s: float, small_diff: float = 1.0e-5
    ) -> (float, float):  # scale
        rsqr = residual @ residual
        r = math.sqrt(rsqr)
        if self.base.numeric_derivs_influence:
            rho_n = self.base.param_instance.influence_func_instance.rho(
                (r - small_diff) ** 2.0, s
            )  # scale
            rho_p = self.base.param_instance.influence_func_instance.rho(
                (r + small_diff) ** 2.0, s
            )  # scale
            rho_deriv = 0.5 * (rho_p - rho_n) / small_diff
            rhop = rho_deriv / r
            rho_c = self.base.param_instance.influence_func_instance.rho(
                r * r, s
            )  # scale
            rho_2nd_deriv = (rho_n + rho_p - 2.0 * rho_c) / (small_diff * small_diff)
            Bterm = (r * rho_2nd_deriv - rho_deriv) / (r * r * r)
        else:
            rhop = self.base.param_instance.influence_func_instance.rhop(
                rsqr, 1.0
            )  # scale
            Bterm = self.base.param_instance.influence_func_instance.Bterm(
                rsqr, 1.0
            )  # scale

        return rhop, Bterm

    def weighted_derivs(
        self, model, lambda_val: float, weight=None, model_ref=None
    ) -> (np.array, np.array):
        if weight is None:
            weight = self.base.weight

        atot = np.zeros(len(model))
        AlBtot = np.zeros((len(model), len(model)))
        small_diff = 1.0e-5  # in case numerical differentiation is specified
        if self.base.scale is None:
            for d, w in zip(self.base.data, weight, strict=True):
                residual, residual_gradient, grad = self.base.calc_residual_derivatives(
                    model, d, model_ref=model_ref, small_diff=small_diff
                )
                rhop, Bterm = self.calc_influence_func_derivatives(
                    residual, 1.0, small_diff=small_diff
                )
                atot += w * rhop * grad
                AlBtot += w * (
                    rhop * np.matmul(np.transpose(residual_gradient), residual_gradient)
                    + lambda_val * Bterm * np.outer(grad, grad)
                )

            return atot, AlBtot
        else:
            for d, w, s in zip(self.base.data, weight, self.base.scale, strict=True):
                residual, residual_gradient, grad = self.base.calc_residual_derivatives(
                    model, d, model_ref=model_ref, small_diff=small_diff
                )
                rhop, Bterm = self.calc_influence_func_derivatives(
                    residual, s, small_diff=small_diff
                )
                atot += w * rhop * grad
                AlBtot += w * (
                    rhop * np.matmul(np.transpose(residual_gradient), residual_gradient)
                    + lambda_val * Bterm * np.outer(grad, grad)
                )

            return atot, AlBtot

    def run(self):
        self.base.param_instance.reset()
        lambda_val = self.lambda_start
        model, model_ref = self.base.init_model()
        last_tot = self.base.objective_func(model, model_ref=model_ref)
        update_model_ref = getattr(self.base.model_instance, "update_model_ref", None)
        if self.base.print_warnings:
            a, AlB = self.weighted_derivs(model, lambda_val, model_ref=model_ref)
            print("Initial model=", model)
            print("Initial model_ref=", model_ref)
            print(
                "Initial params=",
                self.base.param_instance.influence_func_instance.summary(),
            )
            print(
                "Initial tot=",
                last_tot,
                "grad=",
                a,
                "diff_thres=",
                self.base.diff_thres,
            )
            print("Initial weighted derivative lambda_val=", lambda_val, ":")
            print("Initial AlB=", AlB)

        if self.base.debug:
            diffs = []
            model_list = []

        all_good = True
        for itn in range(self.base.max_niterations):
            model_old = model.copy()
            model_refOld = model_ref
            a, AlB = self.weighted_derivs(model_old, lambda_val, model_ref=model_ref)
            try:
                at = np.linalg.solve(AlB, a)
            except np.linalg.LinAlgError:
                lambda_val /= self.lambda_scale
                if self.base.print_warnings:
                    print(
                        "Weighted derivative matrix is singular - rejecting new lambda_val=",
                        lambda_val,
                    )

                continue

            model -= at
            if callable(update_model_ref):
                model_ref = update_model_ref(model, model_ref)

            if (
                self.base.param_instance.at_final_stage()
                and self.base.diff_thres is not None
            ):
                model_max_diff = np.linalg.norm(at, ord=np.inf)
                if self.base.debug is True and model_max_diff > 0.0:
                    diffs.append(math.log10(model_max_diff))

                if model_max_diff < self.base.diff_thres:
                    if self.base.print_warnings:
                        print("Difference threshold reached")

                    break

            tot = self.base.objective_func(model, model_ref=model_ref)
            if self.base.print_warnings:
                print(
                    "itn=",
                    itn,
                    "model=",
                    model,
                    "model_old=",
                    model_old,
                    "params=",
                    self.base.param_instance.influence_func_instance.summary(),
                    "tot=",
                    tot,
                )

            if (
                lambda_val != 0.0
                and self.base.param_instance.influence_func_instance.objective_func_sign()
                * (tot - last_tot)
                > self.residual_tolerance
            ):
                model = model_old
                model_ref = model_refOld
                lambda_val /= self.lambda_scale
                last_tot = self.base.objective_func(model, model_ref=model_ref)
                if self.base.print_warnings:
                    print(
                        "Reject new lambda_val=",
                        lambda_val,
                        "diff=",
                        last_tot - tot,
                        "reverting to model",
                        model,
                    )
            else:
                lambda_val = min(self.lambda_scale * lambda_val, self.lambda_max)
                self.base.param_instance.update()
                if self.base.param_instance.at_final_stage():
                    last_tot = tot
                else:
                    last_tot = self.base.objective_func(model, model_ref=model_ref)

                if self.base.print_warnings:
                    if lambda_val != 0.0 and self.lambda_scale > 1.0:
                        print("Accept new lambda_val=", lambda_val)

            if self.base.debug:
                model_list.append(
                    (
                        itn / (self.base.max_niterations - 1),  # alpha
                        np.copy(model),
                    )
                )

        self.base.param_instance.reset(
            False
        )  # finish with parameters in correct final model
        if self.base.debug:
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
