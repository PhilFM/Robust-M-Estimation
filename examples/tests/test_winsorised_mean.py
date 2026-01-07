import sys
import pytest
import numpy as np

sys.path.append("../src/mean")
from winsorised_mean import winsorised_mean

def test_answer():
    np.random.seed(0) # We want the numbers to be the same on each run
    for test_idx in range(10):
        n_values = np.random.randint(5, 20)
        trim_size = np.random.randint(1,10)
        data = np.zeros((n_values+2*trim_size,1))
        weight = np.zeros(n_values+2*trim_size)

        # calculate the minimum and maximum of random values so that we can
        # construct an array where the winsorised mean is the same as
        # the normal mean without trimmed values
        x_min = 1.0
        x_max = 0.0
        for i in range(n_values):
            data[i][0] = np.random.rand()
            x_min = min(x_min, data[i][0])
            x_max = max(x_max, data[i][0])

        # add the trimmings, trim_size values below the minimum and
        # above the maximum
        for i in range(trim_size):
            data[n_values+i] = -np.random.rand()
            data[n_values+trim_size+i] = 1.0+np.random.rand()

        weight[:] = 1.0

        # calculate winsorised mean a different way
        data_copy = np.copy(data)
        data_copy[n_values:n_values+trim_size] = x_min
        data_copy[n_values+trim_size:n_values+2*trim_size] = x_max

        m_gt = np.mean(data_copy)

        m = winsorised_mean(data, weight, trim_size)
        assert(m[0] == pytest.approx(m_gt))

if __name__ == "__main__":
    test_answer()
