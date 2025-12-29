import numpy as np
import matplotlib.pyplot as plt
import random
import os

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.irls import IRLS
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.gnc_irls_p_params import GNC_IRLSpParams
from gnc_smoothie_philfm.gnc_null_params import GNC_NullParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie_philfm.pseudo_huber_influence_func import PseudoHuberInfluenceFunc
from gnc_smoothie_philfm.gnc_irls_p_influence_func import GNC_IRLSpInfluenceFunc
from gnc_smoothie_philfm.plt_alg_vis import gncs_draw_curve

from gncs_robust_mean import RobustMean

def plot_differences(diffs_welsch_sup_gn, diff_alpha_welsch_sup_gn,
                     diffs_welsch_irls, diff_alpha_welsch_irls,
                     diffs_pseudo_huber_sup_gn, diff_alpha_pseudo_huber_sup_gn,
                     diffs_pseudo_huber_irls, diff_alpha_pseudo_huber_irls,
                     diffs_gnc_irls_p0, diff_alpha_gnc_irls_p0,
                     diffs_gnc_irls_p1, diff_alpha_gnc_irls_p1,
                     test_idx: int, output_folder:str, test_run:bool):
    if not test_run:
        print("diffs_welsch_sup_gn:",diffs_welsch_sup_gn)
        print("diffs_welsch_irls:",diffs_welsch_irls)

    plt.close("all")
    plt.figure(num=1, dpi=240)
    plt.clf()
    ax = plt.gca()
    ax.set_xlim(0,int(max(len(diffs_welsch_sup_gn),len(diffs_welsch_irls),len(diffs_pseudo_huber_sup_gn),len(diffs_pseudo_huber_irls),len(diffs_gnc_irls_p0),len(diffs_gnc_irls_p1))))

    idx = np.argmax(diff_alpha_welsch_sup_gn)
    if idx > 0:
        gncs_draw_curve(plt, diffs_welsch_sup_gn[0:idx+1], ("SupGN", "Welsch", "GNC_Welsch"),
                        lw=0.2, xvalues = np.arange(0,idx+1), add_label=False, markersize=1.0)

    gncs_draw_curve(plt, diffs_welsch_sup_gn[idx:], ("SupGN", "Welsch", "GNC_Welsch"),
                    xvalues = np.arange(idx,len(diffs_welsch_sup_gn)))
    #gncs_draw_curve(plt, diffs_pseudo_huber_sup_gn, ("SupGN", "PseudoHuber", "Welsch"))
    idx = np.argmax(diff_alpha_welsch_irls)
    if idx > 0:
        gncs_draw_curve(plt, diffs_welsch_irls[0:idx+1], ("IRLS", "Welsch", "GNC_Welsch"),
                        lw=0.2, xvalues = np.arange(0,idx+1), add_label=False, markersize=1.0)

    gncs_draw_curve(plt, diffs_welsch_irls[idx:], ("IRLS",  "Welsch", "GNC_Welsch"),
                    xvalues = np.arange(idx,len(diffs_welsch_irls)))
    #gncs_draw_curve(plt, diffs_pseudo_huber_irls,  ("IRLS",  "PseudoHuber", "Welsch"))
    #gncs_draw_curve(plt, diffs_gnc_irls_p1,  ("IRLS",  "GNC_IRLSp",   "GNC_IRLSp1"))
    #gncs_draw_curve(plt, diffs_gnc_irls_p0,  ("IRLS",  "GNC_IRLSp",   "GNC_IRLSp0"))

    ax.set_xlabel(r'Iteration count' )
    ax.set_ylabel(r'log(difference)')
    #plt.box(False)
    #ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    #ax.set_xlim(studentTDOFList[0],studentTDOFList[len(studentTDOFList)-1])
    #ax.set_ylim(0.0,1.1)

    plt.legend()
    plt.savefig(os.path.join(output_folder, "mean_convergence_speed_" + str(test_idx+1) + ".png"), bbox_inches="tight")
    if not test_run:
        plt.show()
    
def randomM11() -> float:
    return 2.0*(np.random.rand()-0.5)

def main(test_run:bool, output_folder:str="../../output"):
    random.seed(0) # We want the numbers to be the same on each run
    with_gnc = True
    n = 1000
    sigma_pop = 1.0
    outlier_fraction = 0.3
    n0 = int((1.0-outlier_fraction)*n+0.5)
    x_range = 6.0

    for test_idx in range(0,4):
        data = np.zeros((n,1))
        mgt = 4.0
        for i in range(n0):
            data[i] = [random.gauss(mgt, sigma_pop)]

        for i in range(n-n0):
            data[n0+i] = x_range*randomM11()

        diff_thres = 1.e-13
        num_sigma_steps = 10
        max_niterations = 200
        residual_tolerance = 1.0e-8
        
        welsch_p = 0.666667
        welsch_sigma = sigma_pop/welsch_p
        welsch_sigma_limit = max(data) - min(data) if with_gnc else welsch_sigma

        model_start = [mgt+0.5]

        model_instance = RobustMean()
    
        welschParamInstance = GNC_WelschParams(WelschInfluenceFunc(), welsch_sigma, welsch_sigma_limit, num_sigma_steps)
        sup_gn_instance = SupGaussNewton(welschParamInstance, model_instance, data,
                                         max_niterations=max_niterations, residual_tolerance=residual_tolerance,
                                         lambda_start=1.0, lambda_scale=1.0, diff_thres=diff_thres,
                                         print_warnings=False,
                                         model_start = None if with_gnc else model_start,
                                         debug=True)
        if sup_gn_instance.run():
            m = sup_gn_instance.final_model
            n_iterations = sup_gn_instance.debug_n_iterations
            diffs_welsch_sup_gn = sup_gn_instance.debug_diffs
            diff_alpha_welsch_sup_gn = sup_gn_instance.debug_diff_alpha
            if not test_run:
                print("GNC Welsch SUP-GN recovered m=",m,"n_iterations=",n_iterations)
                print("GNC Welsch SUP-GN mdiff=",m-mgt)
                print("GNC Welsch SUP-GN diffs=",diffs_welsch_sup_gn)
                print("GNC Welsch SUP-GN diff alpha=",diff_alpha_welsch_sup_gn)
                print("GNC Welsch SUP-GN times weighted_derivs",sup_gn_instance.debug_weighted_derivs_time,"solve",sup_gn_instance.debug_solve_time,"total",sup_gn_instance.debug_total_time)

        irls_instance = IRLS(welschParamInstance, model_instance, data,
                             max_niterations=max_niterations, diff_thres=diff_thres,
                             print_warnings=False,
                             model_start = None if with_gnc else model_start,
                             debug=True)
        if irls_instance.run():
            m = irls_instance.final_model
            n_iterations = irls_instance.debug_n_iterations
            diffs_welsch_irls = irls_instance.debug_diffs
            diff_alpha_welsch_irls = irls_instance.debug_diff_alpha
            if not test_run:
                print("GNC Welsch IRLS recovered m=",m,"n_iterations=",n_iterations)
                print("GNC Welsch IRLS mdiff=",m-mgt)
                print("GNC Welsch IRLS diffs=",diffs_welsch_irls)
                print("GNC Welsch IRLS diff alpha=",diff_alpha_welsch_irls)
                print("GNC Welsch IRLS times update_weights",irls_instance.debug_update_weights_time,"weighted_fit",irls_instance.debug_weighted_fit_time,"total",irls_instance.debug_total_time)

        pseudoHuberParamInstance = GNC_NullParams(PseudoHuberInfluenceFunc(sigma=welsch_sigma))
        sup_gn_instance = SupGaussNewton(pseudoHuberParamInstance, model_instance, data,
                                         max_niterations=max_niterations, residual_tolerance=residual_tolerance,
                                         lambda_start=1.0, lambda_scale=1.0, diff_thres=diff_thres,
                                         print_warnings=False,
                                         model_start = None if with_gnc else model_start,
                                         debug=True)
        if sup_gn_instance.run():
            m = sup_gn_instance.final_model
            n_iterations = sup_gn_instance.debug_n_iterations
            diffs_pseudo_huber_sup_gn = sup_gn_instance.debug_diffs
            diff_alpha_pseudo_huber_sup_gn = sup_gn_instance.debug_diff_alpha
            if not test_run:
                print("Pseudo-Huber G-N recovered m=",m,"n_iterations=",n_iterations)
                print("Pseudo-Huber G-N mdiff=",m-mgt)
                print("Pseudo-Huber G-N diffs=",diffs_pseudo_huber_sup_gn)
                print("Pseudo-Huber G-N diff alpha=",diff_alpha_pseudo_huber_sup_gn)
                print("Pseudo-Huber G-N times weighted_derivs",sup_gn_instance.debug_weighted_derivs_time,"solve",sup_gn_instance.debug_solve_time,"total",sup_gn_instance.debug_total_time)

        irls_instance = IRLS(pseudoHuberParamInstance, model_instance, data,
                             max_niterations=max_niterations, diff_thres=diff_thres,
                             print_warnings=False,
                             model_start = None if with_gnc else model_start,
                             debug=True)
        if irls_instance.run():
            m = irls_instance.final_model
            n_iterations = irls_instance.debug_n_iterations
            diffs_pseudo_huber_irls = irls_instance.debug_diffs
            diff_alpha_pseudo_huber_irls = irls_instance.debug_diff_alpha
            if not test_run:
                print("Pseudo-Huber IRLS recovered m=",m,"n_iterations=",n_iterations)
                print("Pseudo-Huber IRLS mdiff=",m-mgt)
                print("Pseudo-Huber IRLS diffs=",diffs_pseudo_huber_irls)
                print("Pseudo-Huber IRLS diff alpha=",diff_alpha_pseudo_huber_irls)
                print("Pseudo-Huber IRLS times update_weights",irls_instance.debug_update_weights_time,"weighted_fit",irls_instance.debug_weighted_fit_time,"total",irls_instance.debug_total_time)

        gncIrlsp_rscale = 1.0
        gncIrlsp_sigma_base = sigma_pop
        gncIrlsp_epsilon_base = gncIrlsp_rscale*gncIrlsp_sigma_base
        gncIrlsp_epsilon_limit = gncIrlsp_epsilon_base #gncIrlsp_rscale*gnsIrlsp_sigma_limit
        gncIrlsp_beta = 0.8 #math.exp((math.log(gncIrlsp_sigma_base) - math.log(gncIrlsp_sigma_limit))/(num_sigma_steps - 1.0))
        gncIrlspParamInstance = GNC_IRLSpParams(GNC_IRLSpInfluenceFunc(),
                                                0.0, gncIrlsp_rscale, gncIrlsp_epsilon_base, gncIrlsp_epsilon_limit, gncIrlsp_beta)
        irls_instance = IRLS(gncIrlspParamInstance, model_instance, data,
                             max_niterations=max_niterations, diff_thres=diff_thres,
                             print_warnings=False,
                             model_start = None if with_gnc else model_start,
                             debug=True)
        if irls_instance.run():
            m = irls_instance.final_model
            n_iterations = irls_instance.debug_n_iterations
            diffs_gnc_irls_p0 = irls_instance.debug_diffs
            diff_alpha_gnc_irls_p0 = irls_instance.debug_diff_alpha
            if not test_run:
                print("GNC IRLS-p0 recovered m=",m,"n_iterations=",n_iterations)
                print("GNC IRLS-p0 mdiff=",m-mgt)
                print("GNC IRLS-p0 diffs=",diffs_gnc_irls_p0)
                print("GNC IRLS-p0 diff alpha=",diff_alpha_gnc_irls_p0)
                print("GNC IRLS-p0 times update_weights",irls_instance.debug_update_weights_time,"weighted_fit",irls_instance.debug_weighted_fit_time,"total",irls_instance.debug_total_time)

        gncIrlspParamInstance.influence_func_instance.p = 1.0
        irls_instance = IRLS(gncIrlspParamInstance, model_instance, data,
                             max_niterations=max_niterations, diff_thres=diff_thres,
                             print_warnings=False,
                             model_start = None if with_gnc else model_start,
                             debug=True)
        if irls_instance.run():
            m = irls_instance.final_model
            n_iterations = irls_instance.debug_n_iterations
            diffs_gnc_irls_p1 = irls_instance.debug_diffs
            diff_alpha_gnc_irls_p1 = irls_instance.debug_diff_alpha
            if not test_run:
                print("GNC IRLS-p1 recovered m=",m,"n_iterations=",n_iterations)
                print("GNC IRLS-p1 mdiff=",m-mgt)
                print("GNC IRLS-p1 diffs=",diffs_gnc_irls_p1)
                print("GNC IRLS-p1 diff alpha=",diff_alpha_gnc_irls_p1)
                print("GNC IRLS-p1 times update_weights",irls_instance.debug_update_weights_time,"weighted_fit",irls_instance.debug_weighted_fit_time,"total",irls_instance.debug_total_time)

        plot_differences(diffs_welsch_sup_gn, diff_alpha_welsch_sup_gn,
                         diffs_welsch_irls, diff_alpha_welsch_irls,
                         diffs_pseudo_huber_sup_gn, diff_alpha_pseudo_huber_sup_gn,
                         diffs_pseudo_huber_irls, diff_alpha_pseudo_huber_irls,
                         diffs_gnc_irls_p0, diff_alpha_gnc_irls_p0,
                         diffs_gnc_irls_p1, diff_alpha_gnc_irls_p1,
                         test_idx, output_folder, test_run)

    if test_run:
        print("convergence_speed OK")

if __name__ == "__main__":
    main(False) # test_run
