# Iteratively Reweighted Least Squares (IRLS) algorithm, from Huber "Robust Estimation of a
# Location Parameter", The Annals of Mathematical Statistics 35(1), 1964.
# Given data and a model to be applied to the data, IRLS iteratively attempts to reduce the
# residuals (errors) in the data by optimising the model parameters, whilst also
# recalculating weights for each data item based on the size of the residual for the data item.
#
# Parameters:
#   param_instance: Defines the GNC schedule to be followed by IRLS. If GNC is not being used then
#                   this can be a NullParams instance. Should have an internal influence_func_instance
#                   that specifies the IRLS influence function to be used. The influence_func_instance
#                   should provide the following method:
#                   * summary(self) -> str:
#                     A string containing the values of the internal parameters.
#
#                   param_instance itself should provide the following methods:
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
#                   * linear_model_size(self) -> int:
#                          Returns the number of parameters in the model if the
#                          model is linear, otherwise 0. In the linear case the IRLS class uses an
#                          internal weighted_fit() function to fit the model to the data with specified
#                          weights, so that the programmer does not have to implement it.
#                   * weighted_fit(self, data, weight, scale=None) -> (np.array, np.array):
#                          If linear_model_size() returns 0, the model is not linear but a closed-form
#                          solution for the best model given the data with weights can be calculated.
#                          This function implements the closed-form solution. If provided, the scale
#                          array indicates that certain data items are less accurate and so have a
#                          scale value > 1, indicating that the influence function for that data item
#                          should be stretched by the given scale factor.
#  data: An array of data items. Each data item is itself an array.
#  weight: An optional array of float weight values for each data item.
#          If not provided, weights are initialised to one
#  scale: An optional array of scale values, indicating that one or more data items are known to
#         have reduced accuracy, i.e. a wider influence function. The scale indicates the stretching
#         to apply to the influence function for that data item.
#  numeric_derivs_influence: Whether to calculate derivatives of the influence function numerically
#                            from a provided rho() function or directly using a provided rhop() function.
#  max_niterations: Maximum number of IRLS iterations to apply before aborting
#  diff_thres: Terminate when successful update changes the model model parameters by less than this value.
#  print_warnings: Whether to print debugging information.
#  model_start: Optional starting value for model model parameters
#  model_ref_start: Optional starting reference parameters for model, e.g. if optimising rotation
#  debug: Whether to return extra debugging data on exit:
#             * The number of iterations actually applied
#             * The norm of the model parameters change at each iteration, as a list of difference values
#             * A list of the model parameters at each iteration
import math
import numpy as np

from .base_irls import BaseIRLS


class IRLS(BaseIRLS):
    def __init__(
        self,
        param_instance,
        model_instance,
        data,
        weight=None,
        scale=None,
        numeric_derivs_influence=False,
        max_niterations=50,
        diff_thres=1.0e-12,
        print_warnings=False,
        model_start=None,
        model_ref_start=None,
        debug=False,
    ):
        BaseIRLS.__init__(
            self,
            param_instance,
            model_instance,
            data,
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

    def __update_weights(self, model, weight, model_ref=None):
        small_diff = 1.0e-5  # in case numerical differentiation is specified
        self._model_instance.cache_model(model, model_ref=model_ref)
        if self._scale is None:
            for i, d in enumerate(self._data):
                weight[i] = self._weight[i] * self.__updated_weight(
                    self._model_instance.residual(d),
                    1.0,
                    small_diff=small_diff,
                )
        else:
            for i, (d, s) in enumerate(
                zip(self._data, self._scale, strict=True)
            ):
                weight[i] = self._weight[i] * self.__updated_weight(
                    self._model_instance.residual(d), s
                )

    def run(self):
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
            diffs = []
            model_list = []

        for itn in range(self._max_niterations):
            self.__update_weights(model, weight, model_ref=model_ref)
            model_old = model
            if self._model_instance.linear_model_size() > 0:
                model = self.weighted_fit(weight)
            else:
                model, model_ref = self._model_instance.weighted_fit(
                    self._data, weight, self._scale
                )

            if self._param_instance.at_final_stage():
                if self._diff_thres is not None:
                    model_max_diff = np.linalg.norm(model - model_old, ord=np.inf)
                    if self._print_warnings:
                        print("model_max_diff=", model_max_diff)

                    if self._debug is True and model_max_diff > 0.0:
                        if self._print_warnings:
                            print("Adding diff model_max_diff", model_max_diff)

                        diffs.append(math.log10(model_max_diff))

                    if model_max_diff < self._diff_thres:
                        if self._print_warnings:
                            print("Difference threshold reached")

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

            self._param_instance.update()
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
                return model, itn + 1, diffs, model_list
            else:
                return model, model_ref, itn + 1, diffs, model_list
        else:
            if model_ref is None:
                return model
            else:
                return model, model_ref
