import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")
    sys.path.append("../../pypi_package/src/gnc_smoothie_philfm/linear_model")
    sys.path.append("../../pypi_package/src/gnc_smoothie_philfm/cython")

from gnc_smoothie_philfm.irls import IRLS
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie_philfm.plt_alg_vis import gncs_draw_curve
from gnc_smoothie_philfm.linear_model.linear_regressor_welsch import LinearRegressorWelsch
from gnc_smoothie_philfm.linear_model.linear_regressor import LinearRegressor

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
    plt.savefig(os.path.join(output_folder, "plane_fit_convergence_speed_" + str(test_idx+1) + ".png"), bbox_inches='tight')
    if not test_run:
        plt.show()

def main(test_run:bool, output_folder:str="../../output"):
    np.random.seed(0) # We want the numbers to be the same on each run
    with_gnc = True
    for test_idx in range(0,4):
        model_gt = [2.0*(np.random.rand()-0.5), 2.0*(np.random.rand()-0.5), 2.0*(np.random.rand()-0.5)]
        n = 10
        n_data_points = n*n
        data = np.zeros([n_data_points,3])
        noiseLevel = 0.4
        outlier_fraction = 0.3
        for i in range(n):
            y = 0.1*i
            for j in range(n):
                x = 0.1*j
                didx = i*n+j
                if didx < (1.0-outlier_fraction)*n_data_points:
                    data[didx] = (x, y, model_gt[0]*x+model_gt[1]*y+model_gt[2]+noiseLevel*2.0*(np.random.rand()-0.5))
                else:
                    # add outlier
                    data[i] = (x, y, 10.0*2.0*(np.random.rand()-0.5))

        #if not test_run:
        #    print("data=",data)

        diff_thres = 1.e-13
        sigma_base = 0.2
        sigma_limit = 10.0 if with_gnc else sigma_base
        num_sigma_steps = 10
        max_niterations = 500
        print_warnings = False

        model_start = [0.0,0.0,0.0]
        for i in range(3):
            model_start[i] = model_gt[i] + 0.02

        plane_fitter = LinearRegressorWelsch(sigma_base, sigma_limit=sigma_limit, num_sigma_steps=num_sigma_steps,
                                             max_niterations=max_niterations, diff_thres=diff_thres,
                                             print_warnings=print_warnings,
                                             model_start = None if with_gnc else model_start,
                                             debug=True)
        if plane_fitter.run(data):
            diffs_welsch_sup_gn = plane_fitter.debug_diffs
            diff_alpha_welsch_sup_gn = np.array(plane_fitter.debug_diff_alpha)

        param_instance = GNC_WelschParams(WelschInfluenceFunc(), sigma_base, sigma_limit, num_sigma_steps)
        model_instance = LinearRegressor(data[0])
        irls_instance = IRLS(param_instance, data, model_instance=model_instance,
                             max_niterations=max_niterations, diff_thres=diff_thres,
                             print_warnings=print_warnings,
                             model_start = None if with_gnc else model_start,
                             debug=True)
        if irls_instance.run():
            diffs_welsch_irls = irls_instance.debug_diffs
            diff_alpha_welsch_irls = np.array(irls_instance.debug_diff_alpha)
    
        plot_differences(diffs_welsch_sup_gn, diff_alpha_welsch_sup_gn,
                         diffs_welsch_irls, diff_alpha_welsch_irls,
                         test_idx, test_run, output_folder)

    if test_run:
        print("plane_fit_convergence_speed OK")

if __name__ == "__main__":
    main(False) # test_run
