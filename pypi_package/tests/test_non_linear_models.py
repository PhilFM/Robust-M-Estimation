import sys
import pytest
import numpy as np

sys.path.append("../src")
from gnc_smoothie.sup_gauss_newton import SupGaussNewton
from gnc_smoothie.gnc_null_params import GNC_NullParams
from gnc_smoothie.quadratic_influence_func import QuadraticInfluenceFunc
from gnc_smoothie.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie.welsch_influence_func import WelschInfluenceFunc

from check_derivs import check_model_derivs, check_derivs
from quadratic_simple_model import QuadraticSimpleModel

def randomM11() -> float:
    return 2.0*(np.random.rand() - 0.5)

def test_model_derivs():
    for test_idx in range(10):
        # ground-truth model
        model = np.array([randomM11(), randomM11(), randomM11()])

        # build data
        n_points = np.random.randint(5, 10)
        data = np.zeros((n_points,2))
        #for i in range(n_points):
        #    data[i][0] = randomM11()
        #    data[i][1] = randomM11()

        # check Python model
        assert(check_model_derivs(QuadraticSimpleModel(), model, data, include_2nd_derivs=True))

def test_optimiser_derivs():
    for test_idx in range(10):
        # ground-truth model
        model = np.array([randomM11(), randomM11(), randomM11()])

        # build data
        n_points = np.random.randint(5, 10)
        data = np.zeros((n_points,2))
        weight = np.ones(n_points)
        #for i in range(n_points):
        #    data[i][0] = randomM11()
        #    data[i][1] = randomM11()

        # check Python model
        assert(check_derivs(SupGaussNewton(GNC_NullParams(QuadraticInfluenceFunc()), model_instance=QuadraticSimpleModel()),
                            model, data, weight=weight, diff_threshold_AlB=2.e-3))

if __name__ == "__main__":
    np.random.seed(43289) # We want the random numbers to be the same on each run
    #test_model_derivs()
    test_optimiser_derivs()
