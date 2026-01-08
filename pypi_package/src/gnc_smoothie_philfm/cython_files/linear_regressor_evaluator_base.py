import numpy as np
import numpy.typing as npt

# Import cython helper.
try:
    from .linear_regressor_weighted_fit import linear_regressor_weighted_fit
except:
    from linear_regressor_weighted_fit import linear_regressor_weighted_fit

class LinearRegressorEvaluatorBase:
    def __init__(self, data_item):
        if data_item.ndim == 2:
            self._rsize = len(data_item)
            self._msize = len(data_item[0])
        else:
            assert(data_item.ndim == 1)
            self._rsize = 1
            self._msize = len(data_item)

    def set_residual_size(self, residual_size: npt.ArrayLike) -> None:
        residual_size[0] = self._rsize

    def weighted_fit(self, data: npt.ArrayLike, weight: npt.ArrayLike, scale: npt.ArrayLike=None) -> np.array:
        if scale is None:
            this_scale = np.zeros(len(data[0]))
            this_scale[:] = 1.0
        else:
            this_scale = scale[0]

        atot = np.zeros((self._rsize,self._msize))
        Atot = np.zeros((self._rsize,self._msize,self._msize))
        linear_regressor_weighted_fit(np.reshape(data[0], (len(data[0]), self._rsize, self._msize)), weight[0], this_scale, atot, Atot)
        model = np.zeros(self._rsize*self._msize)
        for i in range(self._rsize):
            model[i*self._msize:(i+1)*self._msize] = np.linalg.solve(Atot[i], atot[i])

        return model,None
