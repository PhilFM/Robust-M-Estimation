import math
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
import sys
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

from point_registration import PointRegistration

def plotDifferences(diffsGNCWelsch, diffsPseudoHuber, diffsGNCIRLSp0, diffsGNCIRLSp1, testrun:bool, output_folder:str):
    plt.close("all")
    plt.figure(num=1, dpi=240)
    plt.clf()
    ax = plt.gca()
    ax.set_xlim(0,max(len(diffsGNCWelsch),len(diffsPseudoHuber),len(diffsGNCIRLSp0),len(diffsGNCIRLSp1)))

    gncs_draw_curve(plt, diffsGNCWelsch,   ("SupGN", "Welsch",      "GNC_Welsch")     )
    gncs_draw_curve(plt, diffsPseudoHuber, ("SupGN", "PseudoHuber", "Welsch")     )
    gncs_draw_curve(plt, diffsGNCIRLSp0,   ("IRLS",  "GNC_IRLSp",   "GNC_IRLSp0"))
    gncs_draw_curve(plt, diffsGNCIRLSp1,   ("IRLS",  "GNC_IRLSp",   "GNC_IRLSp1"))

    ax.set_xlabel(r'Iteration count' )
    ax.set_ylabel(r'log(max(rotation/translation difference))')
    #plt.box(False)
    #ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    #ax.set_xlim(studentTDOFList[0],studentTDOFList[len(studentTDOFList)-1])
    #ax.set_ylim(0.0,1.1)

    plt.legend()
    plt.savefig(os.path.join(output_folder, "registration-diffs.png"), bbox_inches='tight')
    if not testrun:
        plt.show()
    
def main(testrun:bool, output_folder:str="../../Output"):
    np.random.seed(0) # We want the numbers to be the same on each run
    N = 1000
    outlierRatio = 0.0 #0.5
    k = int(outlierRatio*N+0.5)
    noise_sigma = 0.5 # noise
    noise_bound = 5.54*noise_sigma
    translationBound    = 10.0

    for test_idx in range(0,1):
        t_gt = np.zeros(3)
        t_gt[0] = np.random.normal(0.0, 1.0)
        t_gt[1] = np.random.normal(0.0, 1.0)
        t_gt[2] = np.random.normal(0.0, 1.0)
        t_gt /= np.linalg.norm(t_gt)
        t_gt = (translationBound) * np.random.rand() * t_gt;

        R_gt = Rot.random().as_matrix()
        if not testrun:
            print("Ground truth R=",R_gt,"t=",t_gt)

        data = np.zeros((N,2,3))
        weight = np.zeros(N)

        for i in range(0,N):
            data[i][0][0] = np.random.normal(0.0, 1.0)
            data[i][0][1] = np.random.normal(0.0, 1.0)
            data[i][0][2] = np.random.normal(0.0, 1.0)
            #print("data[i][0]=",data[i][0])
            RX = np.matmul(R_gt,data[i][0])
            #print("RX=",RX)
            RXpt = RX + t_gt
            data[i][1] = RXpt
            data[i][1][0] += noise_sigma*np.random.normal(0.0, 1.0)
            data[i][1][1] += noise_sigma*np.random.normal(0.0, 1.0)
            data[i][1][2] += noise_sigma*np.random.normal(0.0, 1.0)
            weight[i] = 1.0

        # add outliers 
        nrOutliers  = int(N * outlierRatio + 0.5);
        if (N - nrOutliers) < 3:
            error('Point cloud registration requires minimum 3 inlier correspondences.')
        else:
            if nrOutliers > 0:
                if not testrun:
                    print('point cloud registration: random generate',nrOutliers,'outliers')

                for i in range(N-nrOutliers,N):
                    data[i][1][0] = np.random.normal(0.0, 1.0)
                    data[i][1][1] = np.random.normal(0.0, 1.0)
                    data[i][1][2] = np.random.normal(0.0, 1.0)

        diff_thres = 1.e-12
        num_sigma_steps = 10
        max_niterations = 50
        residual_tolerance = 1.0e-8
        print_warnings = False
        
        welsch_p = 0.666667

        Rs = Rot.as_matrix(Rot.from_mrp([0.25*0.0001,0.25*0.0002,0.25*0.0003]))
        if not testrun:
            print("Rs=",Rs)

        model_start = np.zeros(6)
        model_start[3:6] = t_gt
        model_ref_start = R_gt #np.matmul(Rs,R_gt)
        welschParamInstance = GNC_WelschParams(WelschInfluenceFunc(),
                                               noise_sigma/welsch_p, noise_sigma/welsch_p, num_sigma_steps) # sigma_base, sigma_limit, numSigmaStep
        optimiser_instance = SupGaussNewton(welschParamInstance, PointRegistration(), data, weight=weight,
                                           max_niterations=max_niterations, residual_tolerance=residual_tolerance,
                                           lambda_start=1.0, lambda_scale=1.0, diff_thres=diff_thres, print_warnings=print_warnings,
                                           model_start=model_start, model_ref_start=model_ref_start, debug=True)
        model,model_ref,nIterations,diffsGNCWelsch,model_list = optimiser_instance.run()
        if not testrun:
            #print("diffsGNCWelsch=",diffsGNCWelsch)
            print("GNC Welsch recovered R=",model_ref,"t=",model[3:6],"nIterations=",nIterations)
            print("GNC Welsch Rdiff=",model_ref-R_gt)
            print("GNC Welsch tdiff=",model[3:6]-t_gt)

        optimiser_instance = IRLS(NullParams(PseudoHuberInfluenceFunc(noise_sigma/welsch_p)),
                                 PointRegistration(), data, weight=weight,
                                 max_niterations=max_niterations, diff_thres=diff_thres, print_warnings=print_warnings,
                                 model_start=model_start, model_ref_start=model_ref_start, debug=True)
        model,model_ref,nIterations,diffsPseudoHuber,model_list = optimiser_instance.run()
        if not testrun:
            print("Pseudo-Huber recovered R=",model_ref,"t=",model[3:6],"nIterations=",nIterations)
            print("Pseudo-Huber Rdiff=",model_ref-R_gt)
            print("Pseudo-Huber tdiff=",model[3:6]-t_gt)

        gncIrlsp_rscale = 1.0
        gncIrlsp_sigma_base = noise_sigma
        gncIrlsp_epsilon_base = gncIrlsp_rscale*gncIrlsp_sigma_base
        gncIrlsp_beta = 0.8 #math.exp((math.log(gncIrlsp_sigma_base) - math.log(gncIrlsp_sigma_limit))/(num_sigma_steps - 1.0))
        gncIrlspParamInstance = GNC_IRLSpParams(GNC_IRLSpInfluenceFunc(),
                                                0.0, gncIrlsp_rscale, gncIrlsp_epsilon_base, gncIrlsp_epsilon_base, gncIrlsp_beta)
        optimiser_instance = IRLS(gncIrlspParamInstance, PointRegistration(), data, weight=weight,
                                  max_niterations=max_niterations, diff_thres=diff_thres, print_warnings=print_warnings,
                                  model_start=model_start, model_ref_start=model_ref_start, debug=True)
        model,model_ref,nIterations,diffsGNCIRLSp0,model_list = optimiser_instance.run()
        if not testrun:
            print("GNC IRLS-p0 recovered R=",model_ref,"t=",model[3:6],"nIterations=",nIterations)
            print("GNC IRLS-p0 Rdiff=",model_ref-R_gt)
            print("GNC IRLS-p0 tdiff=",model[3:6]-t_gt)

        gncIrlspParamInstance.influence_func_instance.p = 1.0
        optimiser_instance = IRLS(gncIrlspParamInstance, PointRegistration(), data, weight=weight,
                                  max_niterations=max_niterations, diff_thres=diff_thres, print_warnings=print_warnings,
                                  model_start=model_start, model_ref_start=model_ref_start, debug=True)
        model,model_ref,nIterations,diffsGNCIRLSp1,model_list = optimiser_instance.run()
        if not testrun:
            print("GNC IRLS-p1 recovered R=",model_ref,"t=",model[3:6],"nIterations=",nIterations)
            print("GNC IRLS-p1 Rdiff=",model_ref-R_gt)
            print("GNC IRLS-p1 tdiff=",model[3:6]-t_gt)

        plotDifferences(diffsGNCWelsch, diffsPseudoHuber, diffsGNCIRLSp0, diffsGNCIRLSp1, testrun, output_folder)

    if testrun:
        print("registration_solver OK")
