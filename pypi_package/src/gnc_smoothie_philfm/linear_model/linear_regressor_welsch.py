import numpy as np
import numpy.typing as npt
import math
import sys

try:
    from ..gnc_welsch_params import GNC_WelschParams
    from ..welsch_influence_func import WelschInfluenceFunc
    from ..cython_files.linear_regressor_welsch_evaluator import LinearRegressorWelschEvaluator
except:
    sys.path.append("..")
    from gnc_welsch_params import GNC_WelschParams
    from welsch_influence_func import WelschInfluenceFunc
    from cython_files.linear_regressor_welsch_evaluator import LinearRegressorWelschEvaluator

try:
    from .linear_regressor_base import LinearRegressorBase
except:
    from linear_regressor_base import LinearRegressorBase

class LinearRegressorWelsch(LinearRegressorBase):
    def __init__(
            self,
            sigma_base: float,
            sigma_limit: float = 20.0,
            num_sigma_steps: int = 20,
            max_niterations: int = 50,
            lambda_start: float = 1.0,
            lambda_max: float = 1.0,
            lambda_scale: float = 1.2,
            lambda_thres: float = 0.0,
            diff_thres: float = 1.e-10,
            use_slow_version: bool = False,
            model_start: npt.ArrayLike = None,
            print_warnings: bool = False,
            debug: bool = False
            ):
        LinearRegressorBase.__init__(
            self,
            max_niterations=max_niterations,
            lambda_start=lambda_start,
            lambda_max=lambda_max,
            lambda_scale=lambda_scale,
            lambda_thres=lambda_thres,
            diff_thres=diff_thres,
            use_slow_version=use_slow_version,
            model_start=model_start,
            print_warnings=print_warnings,
            debug=debug
        )
        self.__param_instance = GNC_WelschParams(WelschInfluenceFunc(), sigma_base, sigma_limit, num_sigma_steps)

    def run(self,
            data,
            weight: np.array = None,
            scale: np.array = None):
        data = self.convert_data(data)
        evaluator_instance = None if self._use_slow_version else LinearRegressorWelschEvaluator(data[0])
        return self.run_base(data, self.__param_instance, evaluator_instance, weight, scale)
