import numpy as np
import numpy.typing as npt
import math
import sys

try:
    from ..sup_gauss_newton import SupGaussNewton
    from ..gnc_irls_p_params import GNC_IRLSpParams
    from ..gnc_irls_p_influence_func import GNC_IRLSpInfluenceFunc
    from ..cython.linear_regressor_gnc_irls_p_evaluator import LinearRegressorGNC_IRLSpEvaluator
except:
    sys.path.append("..")
    from sup_gauss_newton import SupGaussNewton
    from gnc_gnc_irls_p_params import GNC_IRLSpParams
    from gnc_irls_p_influence_func import GNC_IRLSpInfluenceFunc
    from cython.linear_regressor_gnc_irls_p_evaluator import LinearRegressorGNC_IRLSpEvaluator

try:
    from .linear_regressor import LinearRegressor
except:
    from linear_regressor import LinearRegressor

class LinearRegressorGNC_IRLSp:
    def __init__(
            self,
            p: float,
            rscale: float,
            epsilon_base: float,
            epsilon_limit: float,
            beta: float,
            max_niterations: int = 50,
            diff_thres: float = 1.e-10,
            use_slow_version: bool = False,
            model_start: npt.ArrayLike = None,
            print_warnings: bool = False,
            debug: bool = False
            ):
        self.__p = p
        self.__rscale = rscale
        self.__epsilon_base = epsilon_base
        self.__epsilon_limit = epsilon_limit
        self.__beta = beta
        self.__max_niterations = max_niterations
        self.__diff_thres = diff_thres
        self.__use_slow_version = use_slow_version
        self.__model_start = model_start
        self.__print_warnings = print_warnings
        self.__debug = debug

    def __convert_model(self, model: np.array, data_item) -> (np.array,np.array):
        if data_item.ndim == 2:
            rsize = len(data_item)
            msize = len(data_item[0])
        else:
            assert(data_item.ndim == 1)
            rsize = 1
            msize = len(data_item)

        modelp = model.reshape((rsize,msize))
        return (modelp[:,:msize-1], modelp[:,msize-1:].reshape(rsize))

    def run(self,
            data,
            weight: np.array = None,
            scale: np.array = None):
        
        param_instance = GNC_IRLSpParams(GNC_IRLSpInfluenceFunc(), self.__p, self.__rscale, self.__epsilon_base, self.__epsilon_limit, self.__beta)
        optimiser_instance = SupGaussNewton(param_instance, data,
                                            model_instance=LinearRegressor(data[0]) if self.__use_slow_version else None,
                                            evaluator_instance = None if self.__use_slow_version else LinearRegressorGNC_IRLSpEvaluator(data[0]),
                                            weight=weight, scale=scale,
                                            max_niterations=self.__max_niterations,
                                            diff_thres=self.__diff_thres,
                                            model_start=self.__model_start,
                                            print_warnings=self.__print_warnings,
                                            debug=self.__debug)
        if optimiser_instance.run():
            self.final_coeff,self.final_intercept = self.__convert_model(optimiser_instance.final_model, data[0])
            self.final_weight = optimiser_instance.final_weight
            if self.__debug:
                self.debug_diffs = optimiser_instance.debug_diffs
                self.debug_diff_alpha = optimiser_instance.debug_diff_alpha
                self.debug_model_list = [(self.__convert_model(model[1], data[0])) for model in optimiser_instance.debug_model_list]
                self.debug_weighted_derivs_time = optimiser_instance.debug_weighted_derivs_time
                self.debug_solve_time = optimiser_instance.debug_solve_time
                self.debug_total_time = optimiser_instance.debug_total_time
                self.debug_n_iterations = optimiser_instance.debug_n_iterations

            return True
        else:
            return False
