import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

if __name__ == "__main__":
    import sys
    sys.path.append("../../../pypi_package/src")

from gnc_smoothie.sup_gauss_newton import SupGaussNewton
from gnc_smoothie.irls import IRLS
from gnc_smoothie.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie.plt_alg_vis import gncs_draw_curve

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

def main(test_run:bool, output_folder:str="../../../output"):
    output_folder += "/image_trs"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    np.random.seed(0) # We want the numbers to be the same on each run
    with_gnc = True

    for test_idx in range(0,4):
        model_gt = [2.0*(np.random.rand()-0.5), 2.0*(np.random.rand()-0.5), 2.0*(np.random.rand()-0.5), 2.0*(np.random.rand()-0.5)]
        n = 20
        data = np.zeros((n*n,4))
        outlier_fraction = 0.2
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
        messages_file = None

        model_start = [0,0,0,0]
        for i in range(4):
            model_start[i] = model_gt[i] + 0.2

        model_instance = TRS()

        param_instance = GNC_WelschParams(WelschInfluenceFunc(), sigma_base,
                                          sigma_limit=sigma_limit, num_sigma_steps=num_sigma_steps)
        sup_gn_instance = SupGaussNewton(param_instance, model_instance=model_instance,
                                         max_niterations=max_niterations, diff_thres=diff_thres,
                                         model_start = None if with_gnc else model_start,
                                         messages_file=messages_file,
                                         debug=True,
                                         lambda_start=1.0)
        if sup_gn_instance.fit(data):
            final_model = sup_gn_instance.final_model
            n_iterations = sup_gn_instance.debug_n_iterations
            diffs_welsch_sup_gn = sup_gn_instance.debug_diffs
            diff_alpha_welsch_sup_gn = np.array(sup_gn_instance.debug_diff_alpha)
            if not test_run:
                print("GNC Welsch SUP-GN recovered final model=",final_model,"n_iterations=",n_iterations,"n_iterations_final_stage=",sup_gn_instance.debug_n_iterations_final_stage)
                print("GNC Welsch SUP-GN final model diff=",final_model-model_gt)
                print("GNC Welsch SUP-GN diffs=",diffs_welsch_sup_gn)
                print("GNC Welsch SUP-GN diff alpha=",diff_alpha_welsch_sup_gn)
                print("GNC Welsch SUP-GN times weighted_derivs",sup_gn_instance.debug_weighted_derivs_time,"solve",sup_gn_instance.debug_solve_time,"final stage",sup_gn_instance.debug_final_stage_time,"total",sup_gn_instance.debug_total_time)

        irls_instance = IRLS(param_instance, model_instance=model_instance,
                             max_niterations=max_niterations, diff_thres=diff_thres,
                             model_start = None if with_gnc else model_start,
                             messages_file=messages_file,
                             debug=True)
        irls_instance.fit(data) # this can fail but we don't care in this context
        final_model = irls_instance.final_model
        n_iterations = irls_instance.debug_n_iterations
        diffs_welsch_irls = irls_instance.debug_diffs
        diff_alpha_welsch_irls = np.array(irls_instance.debug_diff_alpha)
        if not test_run:
            print("GNC Welsch IRLS recovered final model=",final_model,"n_iterations=",n_iterations,"n_iterations_final_stage=",irls_instance.debug_n_iterations_final_stage)
            print("GNC Welsch IRLS final model diff=",final_model-model_gt)
            print("GNC Welsch IRLS diffs=",diffs_welsch_irls)
            print("GNC Welsch IRLS diff alpha=",diff_alpha_welsch_irls)
            print("GNC Welsch IRLS times update_weights",irls_instance.debug_update_weights_time,"weighted_fit",irls_instance.debug_weighted_fit_time,"final stage",irls_instance.debug_final_stage_time,"total",irls_instance.debug_total_time)

        plot_differences(diffs_welsch_sup_gn, diff_alpha_welsch_sup_gn,
                         diffs_welsch_irls, diff_alpha_welsch_irls,
                         test_idx, test_run, output_folder)

    if test_run:
        print("trs_convergence_speed OK")

if __name__ == "__main__":
    main(False) # test_run
