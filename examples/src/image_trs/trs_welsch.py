import numpy as np

from gnc_smoothie.sup_gauss_newton import SupGaussNewton
from gnc_smoothie.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie.welsch_influence_func import WelschInfluenceFunc

from trs import TRS

class TRSWelsch:
    def __init__(
            self,
            sigma: float,
            sigma_limit: float = 20.0,
            num_sigma_steps: int = 20,
            max_niterations: int = 50,
            diff_thres: float = 1.e-10,
            messages_file = None,
            debug: bool = False
            ):
        self.__sigma = sigma
        self.__sigma_limit = sigma_limit
        self.__num_sigma_steps = num_sigma_steps
        self.__max_niterations = max_niterations
        self.__diff_thres = diff_thres
        self.__messages_file = messages_file
        self.__debug = debug

    def run(self,
            data,
            weight: np.array = None,
            scale: np.array = None):
        param_instance = GNC_WelschParams(WelschInfluenceFunc(), self.__sigma,
                                          sigma_limit=self.__sigma_limit, num_sigma_steps=self.__num_sigma_steps)
        optimiser_instance = SupGaussNewton(param_instance, data, model_instance=TRS(), weight=weight, scale=scale,
                                            max_niterations=self.__max_niterations,
                                            diff_thres=self.__diff_thres,
                                            messages_file=self.__messages_file,
                                            debug=self.__debug)
        if optimiser_instance.run():
            self.final_trs = optimiser_instance.final_model
            self.final_weight = optimiser_instance.final_weight
            if self.__debug:
                self.debug_model_list = optimiser_instance.debug_model_list
                self.debug_weighted_derivs_time = optimiser_instance.debug_weighted_derivs_time
                self.debug_solve_time = optimiser_instance.debug_solve_time
                self.debug_total_time = optimiser_instance.debug_total_time
                self.debug_n_iterations = optimiser_instance.debug_n_iterations

            return True
        else:
            return False
