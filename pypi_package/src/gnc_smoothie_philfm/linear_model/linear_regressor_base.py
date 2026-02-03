import numpy as np
import sys
from typing import TextIO

try:
    from ..sup_gauss_newton import SupGaussNewton
except ImportError:
    sys.path.append("..")
    from sup_gauss_newton import SupGaussNewton

try:
    from .linear_regressor import LinearRegressor
except ImportError:
    from linear_regressor import LinearRegressor

class LinearRegressorBase:
    def __init__(
            self,
            *,
            max_niterations: int = 50,
            lambda_start: float = 1.0,
            lambda_max: float = 1.0,
            lambda_scale: float = 1.2,
            lambda_thres: float = 0.0,
            diff_thres: float = 1.e-10,
            use_slow_version: bool = False,
            messages_file: TextIO = None,
            debug: bool = False
            ):
        self.__max_niterations = max_niterations
        self.__lambda_start = lambda_start
        self.__lambda_max = lambda_max
        self.__lambda_scale = lambda_scale
        self.__lambda_thres = lambda_thres
        self.__diff_thres = diff_thres
        self._use_slow_version = use_slow_version
        self.__messages_file = messages_file
        self.__debug = debug

    # check for scipy style X/y "training data/target" arguments, and convert to single data array
    def convert_data(self, data):
        self.__data_is_tuple = isinstance(data, tuple)
        if self.__data_is_tuple:
            assert(len(data) == 2) # data_x, data_y
            data_x = data[0]
            data_y = data[1]
            assert(len(data_x) == len(data_y))
            if data_y.ndim == 1:
                data_y = np.reshape(data_y, (len(data_y),1,1))
                rsize = 1
            else:
                assert(data_y.ndim == 2)
                rsize = len(data_y[0])
                data_y = np.reshape(data_y, (len(data_y),rsize,1))

            if data_x.ndim == 1:
                assert(rsize == 1)
                data_x = np.reshape(data_x, (len(data_x),1,1))
            elif data_x.ndim == 2:
                if rsize == 1:
                    data_x = np.reshape(data_x, (len(data_x),1,len(data_x[0])))
                else:
                    assert(rsize == len(data_x[0]))
                    data_x = np.reshape(data_x, (len(data_x),rsize,1,))
            else:
                assert(data_x.ndim == 3)

            return np.concatenate((data_x, data_y), axis=2)
        else:
            return data

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

    def run_base(self,
                 data,
                 param_instance,
                 evaluator_instance, # only set if use_slow_version is False
                 weight: np.array,
                 scale: np.array,
                 model_start):
        optimiser_instance = SupGaussNewton(param_instance, data,
                                            model_instance = LinearRegressor(data[0]) if self._use_slow_version else None,
                                            evaluator_instance = evaluator_instance,
                                            weight=weight, scale=scale,
                                            max_niterations=self.__max_niterations,
                                            lambda_start=self.__lambda_start,
                                            lambda_max=self.__lambda_max,
                                            lambda_scale=self.__lambda_scale,
                                            lambda_thres=self.__lambda_thres,
                                            diff_thres=self.__diff_thres,
                                            messages_file=self.__messages_file,
                                            debug=self.__debug)
        if optimiser_instance.run(model_start=model_start):
            if self.__data_is_tuple:
                self.final_coeff,self.final_intercept = self.__convert_model(optimiser_instance.final_model, data[0])
            else:
                self.final_model = optimiser_instance.final_model

            self.final_weight = optimiser_instance.final_weight
            if self.__debug:
                self.debug_diffs = optimiser_instance.debug_diffs
                self.debug_diff_alpha = optimiser_instance.debug_diff_alpha
                if self.__data_is_tuple:
                    self.debug_model_list = [(self.__convert_model(model[1], data[0])) for model in optimiser_instance.debug_model_list]
                else:
                    self.debug_model_list = optimiser_instance.debug_model_list

                self.debug_weighted_derivs_time = optimiser_instance.debug_weighted_derivs_time
                self.debug_solve_time = optimiser_instance.debug_solve_time
                self.debug_total_time = optimiser_instance.debug_total_time
                self.debug_n_iterations = optimiser_instance.debug_n_iterations

            return True
        else:
            return False
