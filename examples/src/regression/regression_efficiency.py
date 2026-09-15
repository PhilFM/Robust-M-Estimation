import numpy as np
import matplotlib.pyplot as plt
import os
import math
from pathlib import Path
import argparse

if __name__ == "__main__":
    import sys
    sys.path.append("../../../pypi_package/src")

from gnc_smoothie.plt_alg_vis import gncs_draw_curve

from regression_apply import apply_to_data

def main(test_idx:int = -1, test_run:bool=False, output_folder:str="../../../output", quick_run:bool=False):
    np.set_printoptions(linewidth=150)
    output_folder += "/regression/efficiency"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    output_folder_efficiency = output_folder + "/efficiency"
    Path(output_folder_efficiency).mkdir(parents=True, exist_ok=True)
    output_folder_av_time = output_folder + "/av_time"
    Path(output_folder_av_time).mkdir(parents=True, exist_ok=True)
    output_folder_scale_error = output_folder + "/scale_error"
    Path(output_folder_scale_error).mkdir(parents=True, exist_ok=True)

    ndim = 5 # number of data dimensions
    sigma_pop = 0.1
    welsch_q = 0.66666667
    huber_sigma_scale = 0.6
    gnc_irls_p_epsilon_scale = 1.4

    # number of samples used for statistics
    n_samples_base = 100 if quick_run else 2000
    min_n_samples = 40 if quick_run else 100

    np.random.seed(0) # We want the numbers to be the same on each run

    sample_size_array = [30] if quick_run else [30,100,300,1000]
    outlier_fraction_list = [0.1] if quick_run else [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    if test_idx >= 0:
        tidx = -1

    for x_range in [30.0] if quick_run else [3.0,5.0,10.0,30.0,100.0]:
        for n_data_points in sample_size_array:
            if test_idx >= 0:
                tidx = tidx+1
                if tidx != test_idx:
                    continue

            eff_gnc_welsch_list = []
            eff_huber_list = []
            eff_gnc_irls_p_list = []
            eff_mm_estimation_list = []
            eff_theil_sen_list = []
            eff_ransac_list = []

            av_time_gnc_welsch_list = []
            av_time_huber_list = []
            av_time_gnc_irls_p_list = []
            av_time_mm_estimation_list = []
            av_time_theil_sen_list = []
            av_time_ransac_list = []
            for outlier_fraction in outlier_fraction_list:
                alg_result = apply_to_data(ndim, sigma_pop, welsch_q, huber_sigma_scale, gnc_irls_p_epsilon_scale, x_range, n_data_points, n_samples_base, min_n_samples, outlier_fraction, test_run=test_run)

                #print("   LS variance:",alg_result.var_ls)

                # GNC IRLS Welsch
                fac = np.linalg.cholesky(alg_result.var_gnc_welsch)
                facI = np.linalg.inv(fac)
                eff_mat = np.matmul(facI, np.matmul(alg_result.var_ls, np.matrix.transpose(facI)))
                eff = math.pow(np.linalg.det(eff_mat),1.0/ndim)
                if not test_run:
                    #print("   GNC Welsch estimator variance:",alg_result.var_gnc_welsch)
                    #print("   GNC Welsch estimator eff matrix:",eff_mat)
                    print("   GNC Welsch estimator efficiency: ", eff)

                eff_gnc_welsch_list.append(eff)
                av_time_gnc_welsch_list.append(1000.0*alg_result.av_time_gnc_welsch)
                if not test_run:
                    print("   GNC Welsch estimator average time", 1000.0*alg_result.av_time_gnc_welsch, "msec")

                # Huber
                fac = np.linalg.cholesky(alg_result.var_huber)
                facI = np.linalg.inv(fac)
                eff_mat = np.matmul(facI, np.matmul(alg_result.var_ls, np.matrix.transpose(facI)))
                eff = math.pow(np.linalg.det(eff_mat),1.0/ndim)
                if not test_run:
                    print("   Huber efficiency: ", eff)

                eff_huber_list.append(eff)
                av_time_huber_list.append(1000.0*alg_result.av_time_huber)
                if not test_run:
                    print("   Huber average time", 1000.0*alg_result.av_time_huber, "msec")

                # GNC IRLS-p
                fac = np.linalg.cholesky(alg_result.var_gnc_irls_p)
                facI = np.linalg.inv(fac)
                eff_mat = np.matmul(facI, np.matmul(alg_result.var_ls, np.matrix.transpose(facI)))
                eff = math.pow(np.linalg.det(eff_mat),1.0/ndim)
                if not test_run:
                    print("   GNC IRLS-p efficiency: ", eff)

                eff_gnc_irls_p_list.append(eff)
                av_time_gnc_irls_p_list.append(1000.0*alg_result.av_time_gnc_irls_p)
                if not test_run:
                    print("   GNC IRLS-p efficiency: ", eff)

                # MM estimation
                fac = np.linalg.cholesky(alg_result.var_mm_estimation)
                facI = np.linalg.inv(fac)
                eff_mat = np.matmul(facI, np.matmul(alg_result.var_ls, np.matrix.transpose(facI)))
                eff = math.pow(np.linalg.det(eff_mat),1.0/ndim)
                if not test_run:
                    #print("   MM estimation variance:",alg_result.var_mm_estimation)
                    #print("   MM estimation eff matrix:",eff_mat)
                    print("   MM estimation efficiency: ", eff)

                eff_mm_estimation_list.append(eff)
                av_time_mm_estimation_list.append(1000.0*alg_result.av_time_mm_estimation)
                if not test_run:
                    print("   MM estimation average time", 1000.0*alg_result.av_time_mm_estimation, "msec")

                # Theil-Sen
                fac = np.linalg.cholesky(alg_result.var_theil_sen)
                facI = np.linalg.inv(fac)
                eff_mat = np.matmul(facI, np.matmul(alg_result.var_ls, np.matrix.transpose(facI)))
                eff = math.pow(np.linalg.det(eff_mat),1.0/ndim)
                if not test_run:
                    print("   Theil-Sen efficiency: ", eff)

                eff_theil_sen_list.append(eff)
                av_time_theil_sen_list.append(1000.0*alg_result.av_time_theil_sen)
                if not test_run:
                    print("   Theil-Sen average time", 1000.0*alg_result.av_time_theil_sen, "msec")

                # RANSAC
                fac = np.linalg.cholesky(alg_result.var_ransac)
                facI = np.linalg.inv(fac)
                eff_mat = np.matmul(facI, np.matmul(alg_result.var_ls, np.matrix.transpose(facI)))
                eff = math.pow(np.linalg.det(eff_mat),1.0/ndim)
                if not test_run:
                    print("   RANSAC efficiency: ", eff)

                eff_ransac_list.append(eff)
                av_time_ransac_list.append(1000.0*alg_result.av_time_ransac)
                if not test_run:
                    print("   RANSAC average time", 1000.0*alg_result.av_time_ransac, "msec")

            plt.close("all")
            plt.figure(num=1, dpi=240)
            plt.clf()
            ax = plt.gca()
            gncs_draw_curve(plt, eff_gnc_welsch_list,    ("SupGN",     "Welsch",      "SS"         ), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, eff_huber_list,         ("SupGN",     "PseudoHuber", "PseudoHuber"), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, eff_gnc_irls_p_list,    ("SupGN",     "GNC_IRLSp",   "GNC_IRLSp0" ), xvalues=outlier_fraction_list)

            ax.set_xlabel(r'Outlier fraction' )
            ax.set_ylabel('Relative efficiency')
            ax.set_xlim(0.0,outlier_fraction_list[len(outlier_fraction_list)-1])
            ax.set_ylim(0.0,1.1)

            plt.legend()
            plt.savefig(os.path.join(output_folder_efficiency, "efficiency_known_scale_n" + str(n_data_points) + "_range" + str(int(x_range)) + ".png"), bbox_inches='tight')
            if False: #not test_run:
                plt.show()

            plt.close("all")
            plt.figure(num=1, dpi=240)
            plt.clf()
            ax = plt.gca()
            gncs_draw_curve(plt, av_time_gnc_welsch_list,    ("SupGN",     "Welsch",      "SS"         ), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, av_time_huber_list,         ("SupGN",     "PseudoHuber", "PseudoHuber"), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, av_time_gnc_irls_p_list,    ("SupGN",     "GNC_IRLSp",   "GNC_IRLSp0" ), xvalues=outlier_fraction_list)

            ax.set_xlabel(r'Outlier fraction' )
            ax.set_ylabel(r'Average time in msec')
            ax.set_yscale("log")
            #plt.box(False)
            ax.set_xlim(0.0,outlier_fraction_list[len(outlier_fraction_list)-1])

            plt.legend()
            plt.savefig(os.path.join(output_folder_av_time, "av_time_known_scale_n" + str(n_data_points) + "_range" + str(int(x_range)) + ".png"), bbox_inches='tight')
            if False: #not test_run:
                plt.show()

            plt.close("all")
            plt.figure(num=1, dpi=240)
            plt.clf()
            ax = plt.gca()
            gncs_draw_curve(plt, eff_gnc_welsch_list,     ("SupGN",         "Welsch",         "SS"       ), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, eff_mm_estimation_list,  ("MM-Estimation", "Tukey-Bisquare", "Large-Rho"), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, eff_theil_sen_list,      ("Theil-Sen",     "",               ""         ), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, eff_ransac_list,         ("RANSAC",        "",               ""         ), xvalues=outlier_fraction_list)

            ax.set_xlabel(r'Outlier fraction' )
            ax.set_ylabel('Relative efficiency')
            #plt.box(False)
            ax.set_xlim(0.0,outlier_fraction_list[len(outlier_fraction_list)-1])
            ax.set_ylim(0.0,1.1)

            plt.legend()
            plt.savefig(os.path.join(output_folder_efficiency, "efficiency_n" + str(n_data_points) + "_range" + str(int(x_range)) + ".png"), bbox_inches='tight')
            if False: #not test_run:
                plt.show()

            plt.close("all")
            plt.figure(num=1, dpi=240)
            plt.clf()
            ax = plt.gca()
            gncs_draw_curve(plt, av_time_gnc_welsch_list,     ("SupGN",         "Welsch",         "SS"       ), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, av_time_mm_estimation_list,  ("MM-Estimation", "Tukey-Bisquare", "Large-Rho"), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, av_time_theil_sen_list,      ("Theil-Sen",     "",               ""         ), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, av_time_ransac_list,         ("RANSAC",        "",               ""         ), xvalues=outlier_fraction_list)

            ax.set_xlabel(r'Outlier fraction' )
            ax.set_ylabel(r'Average time in msec')
            ax.set_yscale("log")
            #plt.box(False)
            ax.set_xlim(0.0,outlier_fraction_list[len(outlier_fraction_list)-1])
            #ax.set_ylim(0.0,1.1)

            plt.legend()
            plt.savefig(os.path.join(output_folder_av_time, "av_time_n" + str(n_data_points) + "_range" + str(int(x_range)) + ".png"), bbox_inches='tight')
            if False: #not test_run:
                plt.show()

    if test_run:
        print("regression_efficiency OK")

if __name__ == "__main__":
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-tidx", "--testindex", type=int, required=False, help="test index", default = -1)
    args = vars(ap.parse_args())
    
    main(test_idx=args["testindex"], test_run=False, quick_run=False) # test_run
