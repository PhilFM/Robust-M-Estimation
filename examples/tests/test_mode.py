import sys
import numpy as np

sys.path.append("../src/mean")
from weighted_mode import weighted_mode

def test_mode():
    np.random.seed(100) # We want the numbers to be the same on each run

    # thresholds allows for slightly beyond half the bin size
    half_and_a_bit = 0.501

    for test_idx in range(10):
        n_values = np.random.randint(5, 20)
        x_min = 3.0*np.random.rand()
        x_range = 1.0 + 3.0*np.random.rand()
        m_gt = x_min + np.random.rand()*x_range

        # add some outliers
        outlier_fraction = 0.2
        n0 = int((1.0-outlier_fraction)*n_values+0.5)

        data = np.zeros((n_values,1))
        for i in range(n0):
            data[i][0] = m_gt

        for i in range(n0,n_values):
            data[i][0] = x_min + x_range*np.random.rand()

        bin_size = 0.01
        m = weighted_mode(data, bin_size)
        assert(abs(m-m_gt) < half_and_a_bit*bin_size)

        # add weight
        weight = np.zeros(n_values)
        for i in range(n_values):
            weight[i] = 0.1 + 0.5*np.random.rand()

        m = weighted_mode(data, bin_size, weight=weight)
        assert(abs(m-m_gt) < half_and_a_bit*bin_size)

        # add scale
        scale = np.zeros(n_values)
        for i in range(n_values):
            scale[i] = 0.1 + 0.9*np.random.rand()

        m = weighted_mode(data, bin_size, weight=weight, scale=scale)
        assert(abs(m-m_gt) < half_and_a_bit*bin_size)

if __name__ == "__main__":
    test_mode()
