import numpy as np
import numpy.typing as npt
import math
import sys

try:
    from ..sup_gauss_newton import SupGaussNewton
    from ..gnc_null_params import GNC_NullParams
    from ..pseudo_huber_influence_func import PseudoHuberInfluenceFunc
    from ..cython_files.linear_regressor_pseudo_huber_evaluator import LinearRegressorPseudoHuberEvaluator
except:
    sys.path.append("..")
    from sup_gauss_newton import SupGaussNewton
    from gnc_null_params import GNC_NullParams
    from pseudo_huber_influence_func import PseudoHuberInfluenceFunc
    from cython_files.linear_regressor_pseudo_huber_evaluator import LinearRegressorPseudoHuberEvaluator

try:
    from .linear_regressor import LinearRegressor
    from .linear_regressor_convert import linear_regressor_convert_data, linear_regressor_convert_model
except:
    from linear_regressor import LinearRegressor
    from linear_regressor_convert import linear_regressor_convert_data, linear_regressor_convert_model

class LinearRegressorPseudoHuber:
    def __init__(
            self,
            sigma: float,
            max_niterations: int = 50,
            diff_thres: float = 1.e-10,
            use_slow_version: bool = False,
            model_start: npt.ArrayLike = None,
            print_warnings: bool = False,
            debug: bool = False
            ):
        self.__sigma = sigma
        self.__max_niterations = max_niterations
        self.__diff_thres = diff_thres
        self.__use_slow_version = use_slow_version
        self.__model_start = model_start
        self.__print_warnings = print_warnings
        self.__debug = debug

    def run(self,
            data,
            weight: np.array = None,
            scale: np.array = None):
        
        # check for scipy style X/y "training data/target" arguments, and convert to single data array
        if isinstance(data, tuple):
            data = linear_regressor_convert_data(data)

        param_instance = GNC_NullParams(PseudoHuberInfluenceFunc(self.__sigma))
        optimiser_instance = SupGaussNewton(param_instance, data,
                                            model_instance=LinearRegressor(data[0]) if self.__use_slow_version else None,
                                            evaluator_instance = None if self.__use_slow_version else LinearRegressorPseudoHuberEvaluator(data[0]),
                                            weight=weight, scale=scale,
                                            max_niterations=self.__max_niterations,
                                            diff_thres=self.__diff_thres,
                                            model_start=self.__model_start,
                                            print_warnings=self.__print_warnings,
                                            debug=self.__debug)
        if optimiser_instance.run():
            self.final_coeff,self.final_intercept = linear_regressor_convert_model(optimiser_instance.final_model, data[0])
            self.final_weight = optimiser_instance.final_weight
            if self.__debug:
                self.debug_diffs = optimiser_instance.debug_diffs
                self.debug_diff_alpha = optimiser_instance.debug_diff_alpha
                self.debug_model_list = [(linear_regressor_convert_model(model[1], data[0])) for model in optimiser_instance.debug_model_list]
                self.debug_weighted_derivs_time = optimiser_instance.debug_weighted_derivs_time
                self.debug_solve_time = optimiser_instance.debug_solve_time
                self.debug_total_time = optimiser_instance.debug_total_time
                self.debug_n_iterations = optimiser_instance.debug_n_iterations

            return True
        else:
            return False
