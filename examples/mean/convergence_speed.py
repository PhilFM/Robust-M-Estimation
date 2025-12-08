import numpy as np
import matplotlib.pyplot as plt
import random
import os

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.irls import IRLS
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.gnc_irls_p_params import GNC_IRLSpParams
from gnc_smoothie_philfm.null_params import NullParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie_philfm.pseudo_huber_influence_func import PseudoHuberInfluenceFunc
from gnc_smoothie_philfm.gnc_irls_p_influence_func import GNC_IRLSpInfluenceFunc
from gnc_smoothie_philfm.plt_alg_vis import gncs_draw_curve

from gncs_robust_mean import RobustMean

def plotDifferences(diffsWelschGN, diffAlphaWelschGN,
                    diffsWelschIRLS, diffAlphaWelschIRLS,
                    diffsHuberGN, diffAlphaHuberGN,
                    diffsHuberIRLS, diffAlphaHuberIRLS,
                    diffsGNCIRLSp0, diffAlphaGNCIRLSp0,
                    diffsGNCIRLSp1, diffAlphaGNCIRLSp1,
                    output_folder:str, testrun:bool):
    if not testrun:
        print("diffsWelschGN:",diffsWelschGN)
        print("diffsWelschIRLS:",diffsWelschIRLS)
        print("diffsHuberGN:",diffsHuberGN)
        print("diffsHuberIRLS:",diffsHuberIRLS)

    plt.close("all")
    plt.figure(num=1, dpi=240)
    plt.clf()
    ax = plt.gca()
    ax.set_xlim(0,int(max(len(diffsWelschGN),len(diffsWelschIRLS),len(diffsHuberGN),len(diffsHuberIRLS),len(diffsGNCIRLSp0),len(diffsGNCIRLSp1))))

    gncs_draw_curve(plt, diffsWelschGN,   ("SupGN", "Welsch",      "GNC_Welsch"))
    gncs_draw_curve(plt, diffsHuberGN,    ("SupGN", "PseudoHuber", "Welsch"))
    gncs_draw_curve(plt, diffsWelschIRLS, ("IRLS",  "Welsch",      "GNC_Welsch"))
    gncs_draw_curve(plt, diffsHuberIRLS,  ("IRLS",  "PseudoHuber", "Welsch"))
    gncs_draw_curve(plt, diffsGNCIRLSp1,  ("IRLS",  "GNC_IRLSp",   "GNC_IRLSp1"))
    gncs_draw_curve(plt, diffsGNCIRLSp0,  ("IRLS",  "GNC_IRLSp",   "GNC_IRLSp0"))

    ax.set_xlabel(r'Iteration count' )
    ax.set_ylabel(r'log(difference)')
    #plt.box(False)
    #ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    #ax.set_xlim(studentTDOFList[0],studentTDOFList[len(studentTDOFList)-1])
    #ax.set_ylim(0.0,1.1)

    plt.legend()
    plt.savefig(os.path.join(output_folder, "convergence_speed.png"), bbox_inches="tight")
    if not testrun:
        plt.show()
    
def main(testrun:bool, output_folder:str="../../Output"):
    random.seed(0) # We want the numbers to be the same on each run
    N = 1000
    sigmaPop = 1.0
    noise_sigma = 0.5 # noise

    for test_idx in range(0,1):
        data = np.zeros((N,1))
        weight = np.zeros(N)
        mgt = 3.0
        for i in range(N):
            data[i] = [random.gauss(mgt, sigmaPop)]
            weight[i] = 1.0

        diff_thres = 1.e-13
        num_sigma_steps = 10
        max_niterations = 200
        residual_tolerance = 1.0e-8
        
        welsch_p = 0.666667
        welsch_sigma = noise_sigma/welsch_p
        welsch_sigmaLimit = welsch_sigma #10.0*noise_sigma

        mstart = mgt+0.5 #np.matmul(Rs,R_gt)

        model_instance = RobustMean()
    
        welschParamInstance = GNC_WelschParams(WelschInfluenceFunc(), welsch_sigma, welsch_sigmaLimit, num_sigma_steps)
        sup_gn_instance = SupGaussNewton(welschParamInstance, model_instance, data, weight=weight,
                                         max_niterations=max_niterations, residual_tolerance=residual_tolerance,
                                         lambda_start=1.0, lambda_scale=1.0, diff_thres=diff_thres,
                                         print_warnings=False, model_start=[mstart], debug=True)
        if sup_gn_instance.run():
            m = sup_gn_instance.final_model
            n_iterations = sup_gn_instance.debug_n_iterations
            diffsWelschGN = sup_gn_instance.debug_diffs
            diffAlphaWelschGN = sup_gn_instance.debug_diff_alpha
            if not testrun:
                print("GNC Welsch SUP-GN recovered m=",m,"n_iterations=",n_iterations)
                print("GNC Welsch SUP-GN mdiff=",m-mgt)
                print("GNC Welsch SUP-GN diffs=",diffsWelschGN)
                print("GNC Welsch SUP-GN diff alpha=",diffAlphaWelschGN)

        irls_instance = IRLS(welschParamInstance, model_instance, data, weight=weight,
                             max_niterations=max_niterations, diff_thres=diff_thres,
                             print_warnings=False, model_start=[mstart], debug=True)
        if irls_instance.run():
            m = irls_instance.final_model
            n_iterations = irls_instance.debug_n_iterations
            diffsWelschIRLS = irls_instance.debug_diffs
            diffAlphaWelschIRLS = irls_instance.debug_diff_alpha
            if not testrun:
                print("GNC Welsch IRLS recovered m=",m,"n_iterations=",n_iterations)
                print("GNC Welsch IRLS mdiff=",m-mgt)
                print("GNC Welsch IRLS diffs=",diffsWelschIRLS)
                print("GNC Welsch IRLS diff alpha=",diffAlphaWelschIRLS)

        pseudoHuberParamInstance = NullParams(PseudoHuberInfluenceFunc(sigma=welsch_sigma))
        sup_gn_instance = SupGaussNewton(pseudoHuberParamInstance, model_instance, data, weight=weight,
                                         max_niterations=max_niterations, residual_tolerance=residual_tolerance,
                                         lambda_start=1.0, lambda_scale=1.0, diff_thres=diff_thres,
                                         print_warnings=False, model_start=[mstart], debug=True)
        if sup_gn_instance.run():
            m = sup_gn_instance.final_model
            n_iterations = sup_gn_instance.debug_n_iterations
            diffsHuberGN = sup_gn_instance.debug_diffs
            diffAlphaHuberGN = sup_gn_instance.debug_diff_alpha
            if not testrun:
                print("Pseudo-Huber G-N recovered m=",m,"n_iterations=",n_iterations)
                print("Pseudo-Huber G-N mdiff=",m-mgt)
                print("Pseudo-Huber G-N diffs=",diffsHuberGN)
                print("Pseudo-Huber G-N diff alpha=",diffAlphaHuberGN)

        irls_instance = IRLS(pseudoHuberParamInstance, model_instance, data, weight=weight,
                             max_niterations=max_niterations, diff_thres=diff_thres,
                             print_warnings=False, model_start=[mstart], debug=True)
        if irls_instance.run():
            m = irls_instance.final_model
            n_iterations = irls_instance.debug_n_iterations
            diffsHuberIRLS = irls_instance.debug_diffs
            diffAlphaHuberIRLS = irls_instance.debug_diff_alpha
            if not testrun:
                print("Pseudo-Huber IRLS recovered m=",m,"n_iterations=",n_iterations)
                print("Pseudo-Huber IRLS mdiff=",m-mgt)
                print("Pseudo-Huber IRLS diffs=",diffsHuberIRLS)
                print("Pseudo-Huber IRLS diff alpha=",diffAlphaHuberIRLS)

        gncIrlsp_rscale = 1.0
        gncIrlsp_sigma_base = noise_sigma
        gncIrlsp_epsilon_base = gncIrlsp_rscale*gncIrlsp_sigma_base
        gncIrlsp_epsilon_limit = gncIrlsp_epsilon_base #gncIrlsp_rscale*gnsIrlsp_sigmaLimit
        gncIrlsp_beta = 0.8 #math.exp((math.log(gncIrlsp_sigma_base) - math.log(gncIrlsp_sigmaLimit))/(num_sigma_steps - 1.0))
        gncIrlspParamInstance = GNC_IRLSpParams(GNC_IRLSpInfluenceFunc(),
                                                0.0, gncIrlsp_rscale, gncIrlsp_epsilon_base, gncIrlsp_epsilon_limit, gncIrlsp_beta)
        irls_instance = IRLS(gncIrlspParamInstance, model_instance, data, weight=weight,
                             max_niterations=max_niterations, diff_thres=diff_thres,
                             print_warnings=False, model_start=[mstart], debug=True)
        if irls_instance.run():
            m = irls_instance.final_model
            n_iterations = irls_instance.debug_n_iterations
            diffsGNCIRLSp0 = irls_instance.debug_diffs
            diffAlphaGNCIRLSp0 = irls_instance.debug_diff_alpha
            if not testrun:
                print("GNC IRLS-p0 recovered m=",m,"n_iterations=",n_iterations)
                print("GNC IRLS-p0 mdiff=",m-mgt)
                print("GNC IRLS-p0 diffs=",diffsGNCIRLSp0)
                print("GNC IRLS-p0 diff alpha=",diffAlphaGNCIRLSp0)

        gncIrlspParamInstance.influence_func_instance.p = 1.0
        irls_instance = IRLS(gncIrlspParamInstance, model_instance, data, weight=weight,
                             max_niterations=max_niterations, diff_thres=diff_thres,
                             print_warnings=False, model_start=[mstart], debug=True)
        if irls_instance.run():
            m = irls_instance.final_model
            n_iterations = irls_instance.debug_n_iterations
            diffsGNCIRLSp1 = irls_instance.debug_diffs
            diffAlphaGNCIRLSp1 = irls_instance.debug_diff_alpha
            if not testrun:
                print("GNC IRLS-p1 recovered m=",m,"n_iterations=",n_iterations)
                print("GNC IRLS-p1 mdiff=",m-mgt)
                print("GNC IRLS-p1 diffs=",diffsGNCIRLSp1)
                print("GNC IRLS-p1 diff alpha=",diffAlphaGNCIRLSp1)

        plotDifferences(diffsWelschGN, diffAlphaWelschGN,
                        diffsWelschIRLS, diffAlphaWelschIRLS,
                        diffsHuberGN, diffAlphaHuberGN,
                        diffsHuberIRLS, diffAlphaHuberIRLS,
                        diffsGNCIRLSp0, diffAlphaGNCIRLSp0,
                        diffsGNCIRLSp1, diffAlphaGNCIRLSp1,
                        output_folder, testrun)

    if testrun:
        print("convergence_speed OK")

if __name__ == "__main__":
    main(False) # testrun
