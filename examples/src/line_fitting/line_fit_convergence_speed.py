import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    import sys
    sys.path.append("../../../pypi_package/src")
    sys.path.append("../../../pypi_package/src/gnc_smoothie/linear_model")
    sys.path.append("../../../pypi_package/src/gnc_smoothie/cython_files")

from gnc_smoothie.irls import IRLS
from gnc_smoothie.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie.plt_alg_vis import gncs_draw_curve
from gnc_smoothie.linear_model.linear_regressor_welsch import LinearRegressorWelsch
from gnc_smoothie.linear_model.linear_regressor import LinearRegressor

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
    plt.savefig(os.path.join(output_folder, "line_fit_convergence_speed_" + str(test_idx+1) + ".png"), bbox_inches='tight')
    if not test_run:
        plt.show()

def randomM11() -> float:
    return 2.0*(np.random.rand()-0.5)

def main(test_run:bool, output_folder:str="../../../output"):
    np.random.seed(0) # We want the numbers to be the same on each run
    with_gnc = True
    model_gt = [randomM11(), randomM11()]
    n = 100
    data = np.zeros((n,2))
    sigma_pop = 1.0
    outlier_fraction = 0.3
    n0 = int((1.0-outlier_fraction)*n+0.5)
    y_range = 10.0
    for test_idx in range(0,4):
        for i in range(n):
            x = 0.8*i
            if i < n0:
                data[i] = (x, model_gt[0]*x+model_gt[1] + np.random.normal(0.0, sigma_pop))
            else:
                # add outlier
                data[i] = (x, y_range*randomM11())

        if not test_run:
            print("data=",data)

        diff_thres = 1.e-13
        q = 0.66667
        sigma_base = sigma_pop/q
        sigma_limit = max(data[:,1]) - min(data[:,1]) if with_gnc else sigma_base
        num_sigma_steps = 10
        max_niterations = 100
        messages_file = None

        model_start = [0.0,0.0]
        for i in range(2):
            model_start[i] = model_gt[i] + 0.02

        line_fitter = LinearRegressorWelsch(sigma_base, sigma_limit=sigma_limit, num_sigma_steps=num_sigma_steps,
                                            max_niterations=max_niterations, diff_thres=diff_thres,
                                            messages_file=messages_file, debug=True)
        if line_fitter.run(data, model_start = None if with_gnc else model_start):
            diffs_welsch_sup_gn = line_fitter.debug_diffs
            diff_alpha_welsch_sup_gn = np.array(line_fitter.debug_diff_alpha)

        param_instance = GNC_WelschParams(WelschInfluenceFunc(), sigma_base,
                                          sigma_limit=sigma_limit, num_sigma_steps=num_sigma_steps)
        model_instance = LinearRegressor(data[0])
        irls_instance = IRLS(param_instance, data, model_instance=model_instance,
                             max_niterations=max_niterations, diff_thres=diff_thres,
                             messages_file=messages_file, debug=True)
        if irls_instance.run(model_start = None if with_gnc else model_start):
            diffs_welsch_irls = irls_instance.debug_diffs
            diff_alpha_welsch_irls = np.array(irls_instance.debug_diff_alpha)
    
        plot_differences(diffs_welsch_sup_gn, diff_alpha_welsch_sup_gn,
                         diffs_welsch_irls, diff_alpha_welsch_irls,
                         test_idx, test_run, output_folder)

    if test_run:
        print("line_fit_convergence_speed OK")

if __name__ == "__main__":
    main(False) # test_run
