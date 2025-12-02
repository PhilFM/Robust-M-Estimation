import numpy as np
import sys
import argparse

from RobustMean import RobustMean

sys.path.append("../Library")
from QuadraticInfluenceFunc import QuadraticInfluenceFunc
from WelschInfluenceFunc import WelschInfluenceFunc
from PseudoHuberInfluenceFunc import PseudoHuberInfluenceFunc
from GemanMcClureInfluenceFunc import GemanMcClureInfluenceFunc
from GNC_IRLSpInfluenceFunc import GNC_IRLSpInfluenceFunc
from SupGaussNewton import SupGaussNewton
from NullParams import NullParams
from GNC_IRLSpParams import GNC_IRLSpParams
from check_derivs import check_derivs

def objective_func(m, influence_func_instance):
    return influence_func_instance.objective_func([m])

np.random.seed(0) # We want the numbers to be the same on each run

def main(testrun:bool):
    N = 10
    xgtrange = 10.0
    sigmaPop = 1.0
    xgtborder = 3.0*sigmaPop
    outlierFraction = 0.0
    N0 = int((1.0-outlierFraction)*N+0.5)
    est_p = 0.6666667
    sigma = sigmaPop/est_p

    all_good = True
    for test_idx in range(0,10):
        mgt = np.random.rand()*xgtrange + xgtborder
        data = np.zeros((N,1))
        weight = np.zeros(N)
        for j in range(N0):
            weight[j] = 1.0
            data[j] = [np.random.normal(loc=mgt, scale=sigmaPop)]

        for j in range(N-N0):
            weight[N0+j] = 1.0
            data[N0+j] = [np.random.rand()*(xgtrange + 2.0*xgtborder)]

        x = mgt + 2.0*(np.random.rand()-0.5)*sigmaPop

        print_diffs = False

        if not testrun:
            print("Test number ", 1+test_idx)
            print("  Quadratic:")

        if check_derivs(SupGaussNewton(NullParams(QuadraticInfluenceFunc()), RobustMean(), data, weight=weight), [x],
                        diff_threshold_AlB=1.e-5, print_diffs=print_diffs) is False:
            all_good = False
            break

            print("  Welsch:")

        if check_derivs(SupGaussNewton(NullParams(WelschInfluenceFunc(sigma=sigma)), RobustMean(), data, weight=weight), [x],
                        diff_threshold_AlB=1.e-5, print_diffs=print_diffs) is False:
            all_good = False
            break

        if not testrun:
            print("  PseudoHuber:")

        if check_derivs(SupGaussNewton(NullParams(PseudoHuberInfluenceFunc(sigma=sigma)), RobustMean(), data, weight=weight), [x],
                        diff_threshold_AlB=1.e-5, print_diffs=print_diffs) is False:
            all_good = False
            break

        if not testrun:
            print("  GemanMcClure:")

        if check_derivs(SupGaussNewton(NullParams(GemanMcClureInfluenceFunc(sigma=sigma)), RobustMean(), data, weight=weight), [x],
                        diff_threshold_AlB=1.e-5, print_diffs=print_diffs) is False:
            all_good = False
            break

        if not testrun:
            print("  GNC IRLS-p:")

        p = 1.0
        rscale = 1.0/xgtrange
        epsilon = rscale*sigmaPop
        if check_derivs(SupGaussNewton(NullParams(GNC_IRLSpInfluenceFunc(p=p, rscale=rscale, epsilon=epsilon)), RobustMean(), data, weight=weight), [x],
                        diff_threshold_AlB=1.e-3, print_diffs=print_diffs) is False:
            all_good = False
            break

    if all_good:
        if testrun:
            print("OK")
        else:
            print("ALL DERIVATIVES OK!!")
    else:
        print("Derivative failure")

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--testrun', action="store_true", default=False)
args = parser.parse_args()
main(args.testrun)
