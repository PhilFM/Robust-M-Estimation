import numpy as np
import numpy.typing as npt
import math
import sys

try:
    from ..gnc_null_params import GNC_NullParams
    from ..pseudo_huber_influence_func import PseudoHuberInfluenceFunc
    from ..cython_files.linear_regressor_pseudo_huber_evaluator import LinearRegressorPseudoHuberEvaluator
except:
    sys.path.append("..")
    from gnc_null_params import GNC_NullParams
    from pseudo_huber_influence_func import PseudoHuberInfluenceFunc
    from cython_files.linear_regressor_pseudo_huber_evaluator import LinearRegressorPseudoHuberEvaluator

try:
    from .linear_regressor_base import LinearRegressorBase
except:
    from linear_regressor_base import LinearRegressorBase

class LinearRegressorPseudoHuber(LinearRegressorBase):
    def __init__(
            self,
            sigma: float,
            max_niterations: int = 50,
            lambda_start: float = 1.0,
            lambda_max: float = 1.0,
            lambda_scale: float = 1.2,
            lambda_thres: float = 0.0,
            diff_thres: float = 1.e-10,
            use_slow_version: bool = False,
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
            print_warnings=print_warnings,
            debug=debug
        )
        self.__param_instance = GNC_NullParams(PseudoHuberInfluenceFunc(sigma))

    def run(self,
            data,
            weight: np.array = None,
            scale: np.array = None,
            model_start: npt.ArrayLike = None):
        data = self.convert_data(data)
        evaluator_instance = None if self._use_slow_version else LinearRegressorPseudoHuberEvaluator(data[0])
        return self.run_base(data, self.__param_instance, evaluator_instance, weight, scale, model_start)
