import numpy as np
import matplotlib.pyplot as plt
import os

from line_fit import LineFit

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.irls import IRLS
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie_philfm.plt_alg_vis import gncs_draw_curve

def plotDifferences(diffs_welsch_sup_gn, diff_alpha_welsch_sup_gn,
                    diffs_welsch_irls, diff_alpha_welsch_irls,
                    testrun:bool, output_folder:str):
    if not testrun:
        print("diffs_welsch_sup_gn:",diffs_welsch_sup_gn)
        print("diffs_welsch_irls:",diffs_welsch_irls)

    plt.close("all")
    plt.figure(num=1, dpi=240)
    plt.clf()
    ax = plt.gca()
    ax.set_xlim(0,int(max(len(diffs_welsch_sup_gn),len(diffs_welsch_irls))))

    idx = np.argmax(diff_alpha_welsch_sup_gn)
    gncs_draw_curve(plt, diffs_welsch_sup_gn[0:idx+1], ("SupGN", "Welsch", "GNC_Welsch"), lw=0.2, xvalues = np.arange(0,idx+1))
    gncs_draw_curve(plt, diffs_welsch_sup_gn[idx:],    ("SupGN", "Welsch", "GNC_Welsch"), xvalues = np.arange(idx,len(diffs_welsch_sup_gn)))
    idx = np.argmax(diff_alpha_welsch_irls)
    gncs_draw_curve(plt, diffs_welsch_irls[0:idx+1], ("IRLS",  "Welsch", "GNC_Welsch"), lw=0.2, xvalues = np.arange(0,idx+1))
    gncs_draw_curve(plt, diffs_welsch_irls[idx:],    ("IRLS",  "Welsch", "GNC_Welsch"), xvalues = np.arange(idx,len(diffs_welsch_irls)))

    ax.set_xlabel(r'Iteration count' )
    ax.set_ylabel(r'log(difference)')

    plt.legend()
    plt.savefig(os.path.join(output_folder, "line_fit_convergence_speed.png"), bbox_inches='tight')
    if not testrun:
        plt.show()

def main(testrun:bool, output_folder:str="../../Output"):
    np.random.seed(0) # We want the numbers to be the same on each run
    for test_idx in range(0,10):
        modelGT = [0.2, -2.0]
        N = 10
        data = np.zeros([10,2])
        noiseLevel = 0.4
        outlier_fraction = 0.0
        for i in range(N):
            x = 0.1*i
            if i < (1.0-outlier_fraction)*N:
                data[i] = (x, modelGT[0]*x+modelGT[1] + noiseLevel*2.0*(np.random.rand()-0.5))
            else:
                # add outlier
                data[i] = (x, 10.0*2.0*(np.random.rand()-0.5))

        if not testrun:
            print("data=",data)

        diff_thres = 1.e-13
        sigma_base = 0.2
        sigma_limit = 10.0
        num_sigma_steps = 10
        max_niterations = 100
        print_warnings = False

        model_start = [0.0,0.0]
        for i in range(2):
            model_start[i] = modelGT[i] + 0.02

        param_instance = GNC_WelschParams(WelschInfluenceFunc(), sigma_base, sigma_limit, num_sigma_steps)
        sup_gn_instance = SupGaussNewton(param_instance, LineFit(), data,
                                         max_niterations=max_niterations, diff_thres=diff_thres,
                                         print_warnings=print_warnings, model_start=model_start, debug=True)
        if sup_gn_instance.run():
            diffs_welsch_sup_gn = sup_gn_instance.debug_diffs
            diff_alpha_welsch_sup_gn = np.array(sup_gn_instance.debug_diff_alpha)

        irls_instance = IRLS(param_instance, LineFit(), data,
                             max_niterations=max_niterations, diff_thres=diff_thres,
                             print_warnings=print_warnings, model_start=model_start, debug=True)
        if irls_instance.run():
            diffs_welsch_irls = irls_instance.debug_diffs
            diff_alpha_welsch_irls = np.array(sup_gn_instance.debug_diff_alpha)
    
        plotDifferences(diffs_welsch_sup_gn, diff_alpha_welsch_sup_gn,
                        diffs_welsch_irls, diff_alpha_welsch_irls, testrun, output_folder)

    if testrun:
        print("line_fit_convergence_speed OK")

if __name__ == "__main__":
    main(False) # testrun
