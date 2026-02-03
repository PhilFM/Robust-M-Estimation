import sys
import pytest
import numpy as np

sys.path.append("../src")
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc

def test_answer():
    np.random.seed(0) # We want the numbers to be the same on each run
    for test_idx in range(10):
        sigma_base = np.random.rand()
        sigma_limit = sigma_base + 0.01 + np.random.rand()
        num_sigma_steps = np.random.randint(1, 20)
        influence_func_instance = WelschInfluenceFunc()
        param_instance = GNC_WelschParams(influence_func_instance, sigma_base,
                                          sigma_limit=sigma_limit, num_sigma_steps=num_sigma_steps)

        # ensure smooth GNC transition
        param_instance.reset()
        assert(influence_func_instance.sigma == sigma_limit)
        assert(param_instance.n_steps() == num_sigma_steps)
        last_sigma = influence_func_instance.sigma
        last_alpha = param_instance.alpha()
        assert(last_alpha == 0.0)
        for i in range(num_sigma_steps-1):
            param_instance.increment()
            assert(influence_func_instance.sigma < last_sigma)
            assert(influence_func_instance.sigma > sigma_base)
            assert(param_instance.alpha() > last_alpha)
            assert(param_instance.alpha() < 1.0)
            last_sigma = influence_func_instance.sigma
            last_alpha = param_instance.alpha()

        param_instance.increment()
        assert(influence_func_instance.sigma == sigma_base)
        assert(param_instance.alpha() == 1.0)

if __name__ == "__main__":
    test_answer()
