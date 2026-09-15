import numpy as np
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.append("../../../pypi_package/src")

from gnc_smoothie.sup_gauss_newton import SupGaussNewton
from gnc_smoothie.irls import IRLS
from gnc_smoothie.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie.gnc_irls_p_params import GNC_IRLSpParams
from gnc_smoothie.gnc_null_params import GNC_NullParams
from gnc_smoothie.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie.pseudo_huber_influence_func import PseudoHuberInfluenceFunc
from gnc_smoothie.gnc_irls_p_influence_func import GNC_IRLSpInfluenceFunc
from gnc_smoothie.plt_alg_vis import gncs_draw_curve

from point_registration import PointRegistration

def plot_differences(diffs_gnc_supgn_welsch, diffs_gnc_irls_welsch, diffs_pseudo_huber, diffs_gnc_irls_p0, diffs_gnc_irls_p1,
                     test_idx:int, test_run:bool, output_folder:str):
    plt.close("all")
    plt.figure(num=1, dpi=240)
    plt.clf()
    ax = plt.gca()
    ax.set_xlim(0,max(len(diffs_gnc_supgn_welsch),len(diffs_gnc_irls_welsch))) #len(diffs_pseudo_huber),len(diffs_gnc_irls_p0),len(diffs_gnc_irls_p1)))

    gncs_draw_curve(plt, diffs_gnc_supgn_welsch, ("SupGN", "Welsch",      "GNC_Welsch") )
    gncs_draw_curve(plt, diffs_gnc_irls_welsch,  ("IRLS", "Welsch",      "GNC_Welsch") )
    #gncs_draw_curve(plt, diffs_pseudo_huber,     ("SupGN", "PseudoHuber", "PseudoHuber"))
    #gncs_draw_curve(plt, diffs_gnc_irls_p0,      ("IRLS",  "GNC_IRLSp",   "GNC_IRLSp0") )
    #gncs_draw_curve(plt, diffs_gnc_irls_p1,      ("IRLS",  "GNC_IRLSp",   "GNC_IRLSp1") )

    ax.set_xlabel(r'Iteration count' )
    ax.set_ylabel(r'log(max(rotation/translation difference))')
    #plt.box(False)
    #ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    #ax.set_xlim(studentTDOFList[0],studentTDOFList[len(studentTDOFList)-1])
    #ax.set_ylim(0.0,1.1)

    plt.legend()
    plt.savefig(os.path.join(output_folder, "registration-diffs_" + str(test_idx+1) + ".png"), bbox_inches='tight')
    if not test_run:
        plt.show()
    
def main(test_run:bool, output_folder:str="../../../output"):
    output_folder += "/registration_3d"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    np.random.seed(0) # We want the numbers to be the same on each run
    n_points = 1000
    outlier_ratio = 0.1 #0.5
    noise_sigma = 0.5 # noise
    translation_bound    = 10.0

    for test_idx in range(0,4):
        t_gt = np.zeros(3)
        t_gt[0] = np.random.normal(0.0, 1.0)
        t_gt[1] = np.random.normal(0.0, 1.0)
        t_gt[2] = np.random.normal(0.0, 1.0)
        t_gt /= np.linalg.norm(t_gt)
        t_gt = (translation_bound) * np.random.rand() * t_gt

        R_gt = Rot.random().as_matrix()
        if not test_run:
            print("Ground truth R=",R_gt,"t=",t_gt)

        data = np.zeros((n_points,2,3))
        weight = np.ones(n_points)

        for i in range(0,n_points):
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

        # add outliers 
        n_outliers  = int(n_points * outlier_ratio + 0.5)
        if (n_points - n_outliers) < 3:
            raise ValueError("Point cloud registration requires minimum 3 inlier correspondences")
        else:
            if n_outliers > 0:
                if not test_run:
                    print('point cloud registration: random generate',n_outliers,'outliers')

                for i in range(n_points-n_outliers,n_points):
                    data[i][1][0] = np.random.normal(0.0, 1.0)
                    data[i][1][1] = np.random.normal(0.0, 1.0)
                    data[i][1][2] = np.random.normal(0.0, 1.0)

        diff_thres = 1.e-15
        num_sigma_steps = 10
        max_niterations = 50
        residual_tolerance = 1.0e-8
        messages_file = None
        
        welsch_p = 0.666667

        Rs = Rot.as_matrix(Rot.from_mrp([0.25*0.0001,0.25*0.0002,0.25*0.0003]))
        if not test_run:
            print("Rs=",Rs)

        #model_start = np.zeros(6)
        #model_start[3:6] = t_gt
        #model_ref_start = R_gt #np.matmul(Rs,R_gt)
        welsch_param_instance = GNC_WelschParams(WelschInfluenceFunc(),
                                                 noise_sigma/welsch_p, sigma_limit=noise_sigma/welsch_p,
                                                 num_sigma_steps=num_sigma_steps)
        optimiser_instance = SupGaussNewton(welsch_param_instance, model_instance=PointRegistration(),# numeric_derivs_model=True,
                                            max_niterations=max_niterations, residual_tolerance=residual_tolerance,
                                            lambda_start=1.0, lambda_scale=1.0, diff_thres=diff_thres,
                                            #model_start=model_start, model_ref_start=model_ref_start,
                                            messages_file=messages_file, debug=True)
        if optimiser_instance.fit(data, weight=weight):
            model = optimiser_instance.final_model
            model_ref = optimiser_instance.final_model_ref
            n_iterations = optimiser_instance.debug_n_iterations
            diffs_gnc_supgn_welsch = optimiser_instance.debug_diffs
            if not test_run:
                #print("diffs_gnc_welsch=",diffs_gnc_welsch)
                print("GNC Welsch Sup-GN recovered R=",model_ref,"t=",model[3:6],"n_iterations=",n_iterations)
                print("GNC Welsch Sup-GN Rdiff=",model_ref-R_gt)
                print("GNC Welsch Sup-GN tdiff=",model[3:6]-t_gt)

        irls_instance = IRLS(welsch_param_instance, model_instance=PointRegistration(),
                             max_niterations=max_niterations, diff_thres=diff_thres,
                             #model_start=model_start, model_ref_start=model_ref_start,
                             messages_file=messages_file, debug=True)
        if irls_instance.fit(data, weight=weight):
            model = irls_instance.final_model
            model_ref = irls_instance.final_model_ref
            n_iterations = irls_instance.debug_n_iterations
            diffs_gnc_irls_welsch = irls_instance.debug_diffs
            if not test_run:
                #print("diffs_gnc_welsch=",diffs_gnc_welsch)
                print("GNC Welsch IRLS recovered R=",model_ref,"t=",model[3:6],"n_iterations=",n_iterations)
                print("GNC Welsch IRLS Rdiff=",model_ref-R_gt)
                print("GNC Welsch IRLS tdiff=",model[3:6]-t_gt)

        optimiser_instance = IRLS(GNC_NullParams(PseudoHuberInfluenceFunc(noise_sigma/welsch_p)),
                                  model_instance=PointRegistration(),
                                  max_niterations=max_niterations, diff_thres=diff_thres,
                                  #model_start=model_start, model_ref_start=model_ref_start,
                                  messages_file=messages_file, debug=True)
        if optimiser_instance.fit(data, weight=weight):
            model = optimiser_instance.final_model
            model_ref = optimiser_instance.final_model_ref
            n_iterations = optimiser_instance.debug_n_iterations
            diffs_pseudo_huber = optimiser_instance.debug_diffs
            if not test_run:
                print("Pseudo-Huber recovered R=",model_ref,"t=",model[3:6],"n_iterations=",n_iterations)
                print("Pseudo-Huber Rdiff=",model_ref-R_gt)
                print("Pseudo-Huber tdiff=",model[3:6]-t_gt)

        gnc_irls_p_rscale = 1.0
        gnc_irls_p_sigma_base = noise_sigma
        gnc_irls_p_epsilon_base = gnc_irls_p_rscale*gnc_irls_p_sigma_base
        gnc_irls_p_beta = 0.8 #math.exp((math.log(gnc_irls_p_sigma_base) - math.log(gnc_irls_p_sigma_limit))/(num_sigma_steps - 1.0))
        gnc_irls_p_param_instance = GNC_IRLSpParams(GNC_IRLSpInfluenceFunc(),
                                                   0.0, gnc_irls_p_rscale, gnc_irls_p_epsilon_base,
                                                   epsilon_limit=gnc_irls_p_epsilon_base, beta=gnc_irls_p_beta)
        optimiser_instance = IRLS(gnc_irls_p_param_instance, model_instance=PointRegistration(),
                                  max_niterations=max_niterations, diff_thres=diff_thres,
                                  #model_start=model_start, model_ref_start=model_ref_start,
                                  messages_file=messages_file, debug=True)
        if optimiser_instance.fit(data, weight=weight):
            model = optimiser_instance.final_model
            model_ref = optimiser_instance.final_model_ref
            n_iterations = optimiser_instance.debug_n_iterations
            diffs_gnc_irls_p0 = optimiser_instance.debug_diffs
            if not test_run:
                print("GNC IRLS-p0 recovered R=",model_ref,"t=",model[3:6],"n_iterations=",n_iterations)
                print("GNC IRLS-p0 Rdiff=",model_ref-R_gt)
                print("GNC IRLS-p0 tdiff=",model[3:6]-t_gt)

        gnc_irls_p_param_instance.influence_func_instance.p = 1.0
        optimiser_instance = IRLS(gnc_irls_p_param_instance, model_instance=PointRegistration(),
                                  max_niterations=max_niterations, diff_thres=diff_thres,
                                  #model_start=model_start, model_ref_start=model_ref_start,
                                  messages_file=messages_file, debug=True)
        if optimiser_instance.fit(data, weight=weight):
            model = optimiser_instance.final_model
            model_ref = optimiser_instance.final_model_ref
            n_iterations = optimiser_instance.debug_n_iterations
            diffs_gnc_irls_p1 = optimiser_instance.debug_diffs
            if not test_run:
                print("GNC IRLS-p1 recovered R=",model_ref,"t=",model[3:6],"n_iterations=",n_iterations)
                print("GNC IRLS-p1 Rdiff=",model_ref-R_gt)
                print("GNC IRLS-p1 tdiff=",model[3:6]-t_gt)

        plot_differences(diffs_gnc_supgn_welsch, diffs_gnc_irls_welsch, diffs_pseudo_huber, diffs_gnc_irls_p0, diffs_gnc_irls_p1, test_idx, test_run, output_folder)

    if test_run:
        print("registration_solver OK")

if __name__ == "__main__":
    main(False) # test_run
