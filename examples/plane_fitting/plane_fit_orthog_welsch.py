import numpy as np
import math

from gnc_smoothie_philfm.irls import IRLS
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc

from plane_fit_orthog import PlaneFitOrthog

class PlaneFitOrthogWelsch:
    def __init__(
            self,
            sigma: float,
            sigma_limit: float = 20.0,
            num_sigma_steps: int = 20,
            max_niterations: int = 50,
            diff_thres: float = 1.e-10,
            print_warnings: bool = False,
            debug: bool = False
            ):
        self.__sigma = sigma
        self.__sigma_limit = sigma_limit
        self.__num_sigma_steps = num_sigma_steps
        self.__max_niterations = max_niterations
        self.__diff_thres = diff_thres
        self.__print_warnings = print_warnings
        self.__debug = debug

    def __convert_model(self, model: np.array) -> np.array:
        return model/math.sqrt(model[0]*model[0]+model[1]*model[1]+model[2]*model[2])

    def run(self,
            data,
            weight: np.array = None,
            scale: np.array = None):
        param_instance = GNC_WelschParams(WelschInfluenceFunc(), self.__sigma, self.__sigma_limit, self.__num_sigma_steps)
        optimiser_instance = IRLS(param_instance, PlaneFitOrthog(), data, weight=weight, scale=scale,
                                  max_niterations=self.__max_niterations,
                                  diff_thres=self.__diff_thres,
                                  print_warnings=self.__print_warnings,
                                  debug=self.__debug)
        if optimiser_instance.run():
            self.final_plane = self.__convert_model(optimiser_instance.final_model)
            if self.__debug:
                self.debug_plane_list = [(model[0], self.__convert_model(model[1])) for model in optimiser_instance.debug_model_list]
                self.debug_update_weights_time = optimiser_instance.debug_update_weights_time
                self.debug_weighted_fit_time = optimiser_instance.debug_weighted_fit_time
                self.debug_total_time = optimiser_instance.debug_total_time
                self.debug_n_iterations = optimiser_instance.debug_n_iterations

            return True
        else:
            return False
