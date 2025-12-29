import sys
import pytest
import numpy as np

sys.path.append("../pypi_package/src")
from gnc_smoothie_philfm.gnc_irls_p_params import GNC_IRLSpParams
from gnc_smoothie_philfm.gnc_irls_p_influence_func import GNC_IRLSpInfluenceFunc

def test_answer():
    np.random.seed(0) # We want the numbers to be the same on each run
    for test_idx in range(10):
        p = np.random.rand()
        rscale = 0.5 + np.random.rand()
        epsilon_base = 0.1 + 0.2
        epsilon_limit = 1.0
        beta = 0.95
        influence_func_instance = GNC_IRLSpInfluenceFunc()
        param_instance = GNC_IRLSpParams(influence_func_instance, p, rscale, epsilon_base, epsilon_limit, beta)

        # ensure smooth GNC transition
        param_instance.reset()
        assert(influence_func_instance.epsilon == epsilon_limit)
        assert(param_instance.n_steps() > 0)
        last_epsilon = influence_func_instance.epsilon
        last_alpha = param_instance.alpha()
        assert(last_alpha == 0.0)
        for i in range(param_instance.n_steps()-1):
            param_instance.increment()
            assert(influence_func_instance.epsilon < last_epsilon)
            assert(influence_func_instance.epsilon > epsilon_base)
            assert(param_instance.alpha() > last_alpha)
            assert(param_instance.alpha() < 1.0)
            last_epsilon = influence_func_instance.epsilon
            last_alpha = param_instance.alpha()

        param_instance.increment()
        assert(influence_func_instance.epsilon == epsilon_base)
        assert(param_instance.alpha() == 1.0)

if __name__ == "__main__":
    test_answer()
