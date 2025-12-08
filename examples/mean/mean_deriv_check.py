import numpy as np

from gnc_smoothie_philfm.quadratic_influence_func import QuadraticInfluenceFunc
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie_philfm.pseudo_huber_influence_func import PseudoHuberInfluenceFunc
from gnc_smoothie_philfm.geman_mcclure_influence_func import GemanMcClureInfluenceFunc
from gnc_smoothie_philfm.gnc_irls_p_influence_func import GNC_IRLSpInfluenceFunc
from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.gnc_null_params import GNC_NullParams
from gnc_smoothie_philfm.check_derivs import check_derivs

from gncs_robust_mean import RobustMean

def objective_func(m, influence_func_instance):
    return influence_func_instance.objective_func([m])

np.random.seed(0) # We want the numbers to be the same on each run

def main(testrun:bool, output_folder:str="../../Output"):
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

        if check_derivs(SupGaussNewton(GNC_NullParams(QuadraticInfluenceFunc()), RobustMean(), data, weight=weight), [x],
                        diff_threshold_AlB=1.e-5, print_diffs=print_diffs) is False:
            all_good = False
            break

            print("  Welsch:")

        if check_derivs(SupGaussNewton(GNC_NullParams(WelschInfluenceFunc(sigma=sigma)), RobustMean(), data, weight=weight), [x],
                        diff_threshold_AlB=1.e-5, print_diffs=print_diffs) is False:
            all_good = False
            break

        if not testrun:
            print("  PseudoHuber:")

        if check_derivs(SupGaussNewton(GNC_NullParams(PseudoHuberInfluenceFunc(sigma=sigma)), RobustMean(), data, weight=weight), [x],
                        diff_threshold_AlB=1.e-5, print_diffs=print_diffs) is False:
            all_good = False
            break

        if not testrun:
            print("  GemanMcClure:")

        if check_derivs(SupGaussNewton(GNC_NullParams(GemanMcClureInfluenceFunc(sigma=sigma)), RobustMean(), data, weight=weight), [x],
                        diff_threshold_AlB=1.e-5, print_diffs=print_diffs) is False:
            all_good = False
            break

        if not testrun:
            print("  GNC IRLS-p:")

        p = 1.0
        rscale = 1.0/xgtrange
        epsilon = rscale*sigmaPop
        if check_derivs(SupGaussNewton(GNC_NullParams(GNC_IRLSpInfluenceFunc(p=p, rscale=rscale, epsilon=epsilon)),
                                       RobustMean(), data, weight=weight), [x],
                        diff_threshold_AlB=1.e-3, print_diffs=print_diffs) is False:
            all_good = False
            break

    if all_good:
        if testrun:
            print("mean_deriv_check OK")
        else:
            print("ALL DERIVATIVES OK!!")
    else:
        print("Derivative failure")

if __name__ == "__main__":
    main(False) # testrun
