import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.irls import IRLS
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie_philfm.plt_alg_vis import gncs_draw_curve

from trs import TRS

def plot_differences(diffs_welsch_sup_gn, diff_alpha_welsch_sup_gn,
                     diffs_welsch_irls, diff_alpha_welsch_irls,
                     test_idx: int, test_run:bool, output_folder:str):
    if not test_run:
        print("diffs_welsch_sup_gn:",diffs_welsch_sup_gn)
        print("diffs_welsch_irls:",diffs_welsch_irls)

    plt.close("all")
    plt.figure(num=1, dpi=240)
    plt.clf()
    ax = plt.gca()
    ax.set_xlim(0,int(max(len(diffs_welsch_sup_gn),len(diffs_welsch_irls))))

    idx = np.argmax(diff_alpha_welsch_sup_gn)
    if idx > 0:
        gncs_draw_curve(plt, diffs_welsch_sup_gn[0:idx+1], ("SupGN", "Welsch", "GNC_Welsch"),
                        lw=0.2, xvalues = np.arange(0,idx+1), add_label=False, markersize=1.0)

    gncs_draw_curve(plt, diffs_welsch_sup_gn[idx:], ("SupGN", "Welsch", "GNC_Welsch"),
                    xvalues = np.arange(idx,len(diffs_welsch_sup_gn)))
    idx = np.argmax(diff_alpha_welsch_irls)
    if idx > 0:
        gncs_draw_curve(plt, diffs_welsch_irls[0:idx+1], ("IRLS",  "Welsch", "GNC_Welsch"),
                        lw=0.2, xvalues = np.arange(0,idx+1), add_label=False, markersize=1.0)

    gncs_draw_curve(plt, diffs_welsch_irls[idx:], ("IRLS",  "Welsch", "GNC_Welsch"),
                    xvalues = np.arange(idx,len(diffs_welsch_irls)))

    ax.set_xlabel(r'Iteration count' )
    ax.set_ylabel(r'log(difference)')

    plt.legend()
    plt.savefig(os.path.join(output_folder, "trs_convergence_speed_" + str(test_idx+1) + ".png"), bbox_inches='tight')
    if not test_run:
        plt.show()

def main(test_run:bool, output_folder:str="../../output"):
    np.random.seed(0) # We want the numbers to be the same on each run
    with_gnc = True

    for test_idx in range(0,4):
        model_gt = [2.0*(np.random.rand()-0.5), 2.0*(np.random.rand()-0.5), 2.0*(np.random.rand()-0.5), 2.0*(np.random.rand()-0.5)]
        n = 6
        data = np.zeros((n*n,4))
        outlier_fraction = 0.0
        noise_level = 0.4
        for i in range(n):
            for j in range(n):
                xyi = i*n+j
                data[xyi][0] = j
                data[xyi][1] = i
                if xyi < (1.0-outlier_fraction)*n*n:
                    data[xyi][2] = model_gt[1]*j - model_gt[0]*i + model_gt[2] + noise_level*2.0*(np.random.rand()-0.5)
                    data[xyi][3] = model_gt[0]*j + model_gt[1]*i + model_gt[3] + noise_level*2.0*(np.random.rand()-0.5)
                else:
                    # add outlier
                    data[xyi][2] = 10.0*2.0*(np.random.rand()-0.5)
                    data[xyi][3] = 10.0*2.0*(np.random.rand()-0.5)

        if not test_run:
            print("data=",data)

        diff_thres = 1.e-13
        sigma_base = 0.2
        sigma_limit = 10.0 if with_gnc else sigma_base
        num_sigma_steps = 10
        max_niterations = 100
        print_warnings = False

        model_start = [0,0,0,0]
        for i in range(4):
            model_start[i] = model_gt[i] + 0.2

        model_instance = TRS()

        param_instance = GNC_WelschParams(WelschInfluenceFunc(), sigma_base, sigma_limit, num_sigma_steps)
        sup_gn_instance = SupGaussNewton(param_instance, model_instance, data,
                                         max_niterations=max_niterations, diff_thres=diff_thres,
                                         print_warnings=print_warnings,
                                         model_start = None if with_gnc else model_start,
                                         debug=True,
                                         lambda_start=1.0)
        if sup_gn_instance.run():
            diffs_welsch_sup_gn = sup_gn_instance.debug_diffs
            diff_alpha_welsch_sup_gn = np.array(sup_gn_instance.debug_diff_alpha)

        irls_instance = IRLS(param_instance, model_instance, data,
                             max_niterations=max_niterations, diff_thres=diff_thres,
                             print_warnings=print_warnings,
                             model_start = None if with_gnc else model_start,
                             debug=True)
        irls_instance.run() # this can fail but we don't care in this context
        diffs_welsch_irls = irls_instance.debug_diffs
        diff_alpha_welsch_irls = np.array(sup_gn_instance.debug_diff_alpha)
    
        plot_differences(diffs_welsch_sup_gn, diff_alpha_welsch_sup_gn,
                         diffs_welsch_irls, diff_alpha_welsch_irls,
                         test_idx, test_run, output_folder)

    if test_run:
        print("trs_convergence_speed OK")

if __name__ == "__main__":
    main(False) # test_run
