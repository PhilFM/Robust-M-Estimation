import numpy as np
import numpy.typing as npt

def tukey_trimean(data:npt.ArrayLike) -> np.array:
    quantile = np.quantile(data, [0.25,0.5,0.75])
    return 0.25*(quantile[0] + 2.0*quantile[1] + quantile[2])
