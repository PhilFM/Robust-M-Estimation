import sys
import numpy as np
import math

sys.path.append("../src")
from gnc_smoothie.projective_normalisation import ProjectiveNormalisation

def test_vector_normalisation():
    np.random.seed(0) # We want the numbers to be the same on each run
    for vdim in range(2,5):
        for test_idx in range(2):
            n_points = 100
            data = np.zeros((n_points,vdim))
            sigma_pop = 1.0
            mean_gt = 0.0
            for i in range(n_points):
                for j in range(vdim-1):
                    data[i][j] = np.random.normal(mean_gt, sigma_pop)

                data[i][vdim-1] = -1.0

            pnorm = ProjectiveNormalisation()
            pnorm.run(data)
            assert(pnorm.error_string is None)

            # check that normalisation worked
            tot_s = np.zeros((vdim,vdim))
            for v in pnorm.ndata:
                tot_s += np.outer(v,v)

            # scale to identity
            tot_s *= vdim/n_points
            for i in range(vdim):
                for j in range(vdim):
                    ref_val = 1.0 if i == j else 0.0
                    assert(abs(tot_s[i][j]-ref_val) < 1.e-10)

if __name__ == "__main__":
    test_vector_normalisation()
