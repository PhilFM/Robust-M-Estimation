import sys
import pytest
import numpy as np

sys.path.append("../src/mean")
from trimmed_mean import trimmed_mean

def test_trimmed_mean():
    np.random.seed(0) # We want the numbers to be the same on each run
    for test_idx in range(10):
        n_values = np.random.randint(5, 20)
        trim_size = np.random.randint(1,10)
        data = np.zeros((n_values+2*trim_size,1))
        weight = np.zeros(n_values+2*trim_size)
        for i in range(n_values):
            data[i][0] = np.random.rand()

        m_gt = np.mean(data[0:n_values])

        # add the trimmings, trim_size values below the minimum and
        # above the maximum
        for i in range(trim_size):
            data[n_values+i] = -np.random.rand()
            data[n_values+trim_size+i] = 1.0+np.random.rand()

        weight[:] = 1.0

        m = trimmed_mean(data, trim_size, weight)
        assert(m[0] == pytest.approx(m_gt))

if __name__ == "__main__":
    test_trimmed_mean()
