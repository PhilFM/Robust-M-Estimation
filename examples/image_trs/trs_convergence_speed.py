import numpy as np
import matplotlib.pyplot as plt
import os

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.irls import IRLS
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie_philfm.plt_alg_vis import gncs_draw_curve

from trs import TRS

def plotDifferences(diffsWelschGN, diffsWelschIRLS, testrun:bool, output_folder:str):
    if not testrun:
        print("diffsWelschGN:",diffsWelschGN)
        print("diffsWelschIRLS:",diffsWelschIRLS)

    plt.close("all")
    plt.figure(num=1, dpi=240)
    plt.clf()
    ax = plt.gca()
    ax.set_xlim(0,int(max(len(diffsWelschGN),len(diffsWelschIRLS))))

    gncs_draw_curve(plt, diffsWelschGN,   ("SupGN", "Welsch", "GNC_Welsch"))
    gncs_draw_curve(plt, diffsWelschIRLS, ("IRLS",  "Welsch", "GNC_Welsch"))

    ax.set_xlabel(r'Iteration count' )
    ax.set_ylabel(r'log(difference)')

    plt.legend()
    plt.savefig(os.path.join(output_folder, "trs_convergence_speed.png"), bbox_inches='tight')
    if not testrun:
        plt.show()

def main(testrun:bool, output_folder:str="../../Output"):
    np.random.seed(0) # We want the numbers to be the same on each run

    for test_idx in range(0,10):
        modelGT = [2.0*(np.random.rand()-0.5), 2.0*(np.random.rand()-0.5), 2.0*(np.random.rand()-0.5), 2.0*(np.random.rand()-0.5)]
        N = 6
        data = np.zeros((N*N,4))
        outlier_fraction = 0.0
        noiseLevel = 0.4
        for i in range(N):
            for j in range(N):
                xyi = i*N+j
                data[xyi][0] = j
                data[xyi][1] = i
                if xyi < (1.0-outlier_fraction)*N*N:
                    data[xyi][2] = modelGT[1]*j - modelGT[0]*i + modelGT[2] + noiseLevel*2.0*(np.random.rand()-0.5)
                    data[xyi][3] = modelGT[0]*j + modelGT[1]*i + modelGT[3] + noiseLevel*2.0*(np.random.rand()-0.5)
                else:
                    # add outlier
                    data[xyi][2] = 10.0*2.0*(np.random.rand()-0.5)
                    data[xyi][3] = 10.0*2.0*(np.random.rand()-0.5)

        if not testrun:
            print("data=",data)

        diff_thres = 1.e-13
        sigma_base = 0.2
        sigma_limit = 10.0
        num_sigma_steps = 10
        max_niterations = 100
        print_warnings = False

        model_start = [0,0,0,0]
        for i in range(4):
            model_start[i] = modelGT[i] + 0.2

        model_instance = TRS()

        param_instance = GNC_WelschParams(WelschInfluenceFunc(), sigma_base, sigma_limit, num_sigma_steps)
        sup_gn_instance = SupGaussNewton(param_instance, model_instance, data,
                                         max_niterations=max_niterations, diff_thres=diff_thres,
                                         print_warnings=print_warnings, model_start=model_start, debug=True,
                                         lambda_start=1.0)
        if sup_gn_instance.run():
            diffsWelschGN = sup_gn_instance.debug_diffs

        irls_instance = IRLS(param_instance, model_instance, data,
                             max_niterations=max_niterations, diff_thres=diff_thres,
                             print_warnings=print_warnings, model_start=model_start, debug=True)
        irls_instance.run() # this can fail but we don't care in this context
        diffsWelschIRLS = irls_instance.debug_diffs
    
        plotDifferences(diffsWelschGN, diffsWelschIRLS, testrun, output_folder)

    if testrun:
        print("trs_convergence_speed OK")

if __name__ == "__main__":
    main(False) # testrun
