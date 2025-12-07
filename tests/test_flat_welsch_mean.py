import sys
import pytest
import numpy as np

sys.path.append("../pypi_package/src")
from gnc_smoothie_philfm.base_irls import BaseIRLS
from gnc_smoothie_philfm.null_params import NullParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc

sys.path.append("../examples/mean")
from flat_welsch_mean import flat_welsch_mean
from gncs_robust_mean import RobustMean

def test_answer():
    np.random.seed(0) # We want the numbers to be the same on each run
    for test_idx in range(10):
        n_values = np.random.randint(5, 20)
        data = np.zeros((n_values,1))
        weight = np.zeros(n_values)
        scale = np.zeros(n_values)

        x_range = 5.0+3.0*np.random.rand()
        for i in range(n_values):
            data[i][0] = x_range*np.random.rand()
            weight[i] = 1.0 #np.random.rand()
            scale[i] = 1.0 #+np.random.rand()

        sigma = 0.1+np.random.rand()
        m = flat_welsch_mean(data, sigma, weight, scale)

        # determine ground truth by sampling
        xlist = np.linspace(0.0, x_range, num=30000)
        param_instance = NullParams(WelschInfluenceFunc(sigma))
        optimiser_instance = BaseIRLS(param_instance, RobustMean(), data, weight, scale)
        v_max = 0.0
        x_max = None
        for x in xlist:
            v = optimiser_instance.objective_func([x])
            if v > v_max:
                v_max = v
                x_max = x

        print("m=",m,"x_max=",x_max)
        assert(m == pytest.approx(x_max, 1.e-4))
