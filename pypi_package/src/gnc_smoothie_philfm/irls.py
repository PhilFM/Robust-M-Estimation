import math
import numpy as np
import numpy.typing as npt
from typing import TextIO
import time

from .base_irls import BaseIRLS


class IRLS(BaseIRLS):
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
        numeric_derivs_influence: bool = False,
        max_niterations: int = 50,
        diff_thres: float = 1.0e-12,
        messages_file: TextIO = None,
        model_start: npt.ArrayLike = None,
        model_ref_start=None,
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
            numeric_derivs_influence=numeric_derivs_influence,
            max_niterations=max_niterations,
            diff_thres=diff_thres,
            messages_file=messages_file,
            debug=debug,
        )

    def model_instance(self):
        return self._model_instance()

    def run(self,
            *,
            model_start: npt.ArrayLike = None,
            model_ref_start: npt.ArrayLike=None,
            ) -> bool:
        self._param_instance.reset()
        weight = [None] * self._dsize
        for didx in range(self._dsize):
            if self._data[didx] is not None:
                weight[didx] = np.copy(self._weight[didx])

        model, model_ref = self._init_model(model_start, model_ref_start)
        if self._messages_file is not None:
            print(
                "Initial model=",
                model,
                "params=",
                self._param_instance.influence_func_instance.summary(),
                "diff_thres=",
                self._diff_thres,
                file=self._messages_file
            )

        if self._debug:
            self.debug_diffs = []
            self.debug_diff_alpha = []
            self.debug_model_list = []
            self.debug_model_list.append(
                (
                    0.0,  # alpha
                    np.copy(model),
                )
            )

            self.debug_update_weights_time = 0.0
            self.debug_weighted_fit_time = 0.0
            self.debug_total_time = 0.0
            start_time_total = time.time()

        all_good = False
        for itn in range(self._max_niterations):
            if self._debug:
                start_time = time.time()

            self.update_weights(model, weight, model_ref=model_ref)
            if self._debug:
                self.debug_update_weights_time += time.time() - start_time
                start_time = time.time()

            model_old = model
            if self._evaluator_instance is not None or callable(self._linear_model_size):
                model, model_ref = self.weighted_fit(weight)
            else:
                model, model_ref = self.model_weighted_fit(weight=weight)

            if self._messages_file is not None:
                print("model=",model, file=self._messages_file)

            if self._debug:
                self.debug_weighted_fit_time += time.time() - start_time

            if self._param_instance.alpha() == 1.0:
                if self._diff_thres is not None:
                    model_max_diff = np.linalg.norm(model - model_old, ord=np.inf)
                    if self._messages_file is not None:
                        print("model_max_diff=", model_max_diff, file=self._messages_file)

                    if self._debug is True and model_max_diff > 0.0:
                        if self._messages_file is not None:
                            print("Adding diff model_max_diff", model_max_diff, file=self._messages_file)

                        self.debug_diffs.append(math.log10(model_max_diff))
                        self.debug_diff_alpha.append(self._param_instance.alpha())

                    if model_max_diff < self._diff_thres:
                        if self._messages_file is not None:
                            print("Difference threshold reached", file=self._messages_file)

                        all_good = True
                        break

            if self._messages_file is not None:
                print(
                    "itn=",
                    itn,
                    "model=",
                    model,
                    "params=",
                    self._param_instance.influence_func_instance.summary(),
                    file=self._messages_file
                )

            self._param_instance.increment()
            if self._debug:
                self.debug_model_list.append(
                    (
                        (1 + itn) / (self._max_niterations - 1),  # alpha
                        np.copy(model),
                    )
                )

        self.finalise(model, model_ref=model_ref, weight=weight, itn=itn, total_time = time.time() - start_time_total if self._debug else 0)
        return all_good
