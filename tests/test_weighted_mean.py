import sys
import pytest
import numpy as np

sys.path.append("../examples/mean")
from weighted_mean import weighted_mean

def test_answer():
    np.random.seed(0) # We want the numbers to be the same on each run
    for test_idx in range(10):
        n_values = np.random.randint(5, 20)
        data = np.zeros((n_values,1))
        weight = np.zeros(n_values)
        for i in range(n_values):
            data[i][0] = np.random.rand()
            weight[i] = 1.0

        m = weighted_mean(data, weight)
        m_gt = np.mean(data)
        assert(m[0] == pytest.approx(m_gt))

        scale = np.zeros(n_values)
        scale[:] = 1.0
        m = weighted_mean(data, weight, scale)
        assert(m[0] == pytest.approx(m_gt))
