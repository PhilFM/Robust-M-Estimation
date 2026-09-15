import numpy as np
import matplotlib.pyplot as plt
import os
import math
from pathlib import Path

if __name__ == "__main__":
    import sys
    sys.path.append("../../../pypi_package/src")

from gnc_smoothie.plt_alg_vis import gncs_draw_curve

from line_fit_apply import apply_to_data

def main(test_run:bool, output_folder:str="../../../output", quick_run:bool=False):
    output_folder += "/line_fit/efficiency"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    sigma_pop = 0.01
    welsch_q = 0.66666667
    huber_sigma_scale = 0.6
    gnc_irls_p_epsilon_scale = 1.4

    # number of samples used for statistics
    n_samples_base = 100 if quick_run else 20000
    min_n_samples = 40 if quick_run else 1000

    np.random.seed(0) # We want the numbers to be the same on each run

    show_known_scale_results = True
    outlier_fraction_list = [0.0,0.1,0.2,0.3] if quick_run else [0.0,0.05,0.1,0.15,0.2,0.35,0.3,0.35,0.4]
    for x_range in [5.0,30.0] if quick_run else [3.0,5.0,10.0,30.0,100.0]:
        sample_size_array = [300] if quick_run else [30,100,300,1000]
        for n in sample_size_array:
            if show_known_scale_results:
                eff_gnc_welsch_list = []
                eff_huber_list = []
                eff_gnc_irls_p_list = []

            eff_theil_sen_list = []
            eff_ransac_list = []
            eff_hough_list = []
            eff_ls_list = []

            if show_known_scale_results:
                av_time_gnc_welsch_list = []
                av_time_huber_list = []
                av_time_gnc_irls_p_list = []

            av_time_theil_sen_list = []
            av_time_ransac_list = []
            av_time_hough_list = []
            av_time_ls_list = []
            for outlier_fraction in outlier_fraction_list:
                alg_result = apply_to_data(show_known_scale_results, sigma_pop, welsch_q, huber_sigma_scale, gnc_irls_p_epsilon_scale, x_range, n, n_samples_base, min_n_samples, outlier_fraction, test_run=test_run)

                if show_known_scale_results:
                    # GNC IRLS Welsch
                    fac = np.linalg.cholesky(alg_result.var_gnc_welsch)
                    facI = np.linalg.inv(fac)
                    eff_mat = np.matmul(facI, np.matmul(alg_result.var_predicted, np.matrix.transpose(facI)))
                    eff = math.sqrt(np.linalg.det(eff_mat))
                    if not test_run:
                        print("   GNC Welsch estimator efficiency: ", eff)

                    eff_gnc_welsch_list.append(eff)
                    av_time_gnc_welsch_list.append(1000.0*alg_result.av_time_gnc_welsch)

                    # Huber
                    fac = np.linalg.cholesky(alg_result.var_huber)
                    facI = np.linalg.inv(fac)
                    eff_mat = np.matmul(facI, np.matmul(alg_result.var_predicted, np.matrix.transpose(facI)))
                    eff = math.sqrt(np.linalg.det(eff_mat))
                    if not test_run:
                        print("   Huber efficiency: ", eff)

                    eff_huber_list.append(eff)
                    av_time_huber_list.append(1000.0*alg_result.av_time_huber)

                    # GNC IRLS-p
                    fac = np.linalg.cholesky(alg_result.var_gnc_irls_p)
                    facI = np.linalg.inv(fac)
                    eff_mat = np.matmul(facI, np.matmul(alg_result.var_predicted, np.matrix.transpose(facI)))
                    eff = math.sqrt(np.linalg.det(eff_mat))
                    if not test_run:
                        print("   GNC IRLS-p efficiency: ", eff)

                    eff_gnc_irls_p_list.append(eff)
                    av_time_gnc_irls_p_list.append(1000.0*alg_result.av_time_gnc_irls_p)

                # Theil-Sen
                fac = np.linalg.cholesky(alg_result.var_theil_sen)
                facI = np.linalg.inv(fac)
                eff_mat = np.matmul(facI, np.matmul(alg_result.var_predicted, np.matrix.transpose(facI)))
                eff = math.sqrt(np.linalg.det(eff_mat))
                if not test_run:
                    print("   Theil-Sen efficiency: ", eff)

                eff_theil_sen_list.append(eff)
                av_time_theil_sen_list.append(1000.0*alg_result.av_time_theil_sen)

                # RANSAC
                fac = np.linalg.cholesky(alg_result.var_ransac)
                facI = np.linalg.inv(fac)
                eff_mat = np.matmul(facI, np.matmul(alg_result.var_predicted, np.matrix.transpose(facI)))
                eff = math.sqrt(np.linalg.det(eff_mat))
                if not test_run:
                    print("   RANSAC efficiency: ", eff)

                eff_ransac_list.append(eff)
                av_time_ransac_list.append(1000.0*alg_result.av_time_ransac)

                # Hough transform
                fac = np.linalg.cholesky(alg_result.var_hough)
                facI = np.linalg.inv(fac)
                eff_mat = np.matmul(facI, np.matmul(alg_result.var_predicted, np.matrix.transpose(facI)))
                eff = math.sqrt(np.linalg.det(eff_mat))
                if not test_run:
                    print("   Hough transform efficiency: ", eff)

                eff_hough_list.append(eff)
                av_time_hough_list.append(1000.0*alg_result.av_time_hough)

                # Least squares
                fac = np.linalg.cholesky(alg_result.var_ls)
                facI = np.linalg.inv(fac)
                eff_mat = np.matmul(facI, np.matmul(alg_result.var_predicted, np.matrix.transpose(facI)))
                eff = math.sqrt(np.linalg.det(eff_mat))
                if not test_run:
                    print("   Least squares efficiency: ", eff)

                eff_ls_list.append(eff)
                av_time_ls_list.append(1000.0*alg_result.av_time_ls)

            if show_known_scale_results:
                plt.close("all")
                plt.figure(num=1, dpi=240)
                plt.clf()
                ax = plt.gca()
                gncs_draw_curve(plt, eff_gnc_welsch_list,    ("SupGN",     "Welsch",      "GNC_Welsch" ), xvalues=outlier_fraction_list)
                gncs_draw_curve(plt, eff_huber_list,         ("SupGN",     "PseudoHuber", "PseudoHuber"), xvalues=outlier_fraction_list)
                gncs_draw_curve(plt, eff_gnc_irls_p_list,    ("SupGN",     "GNC_IRLSp",   "GNC_IRLSp0" ), xvalues=outlier_fraction_list)

                ax.set_xlabel(r'Outlier fraction' )
                ax.set_ylabel('Relative efficiency')
                ax.set_xlim(0.0,outlier_fraction_list[len(outlier_fraction_list)-1])
                ax.set_ylim(0.0,1.1)

                plt.legend()
                plt.savefig(os.path.join(output_folder, "efficiency_known_scale_n" + str(n) + "_range" + str(int(x_range)) + ".png"), bbox_inches='tight')
                if False: #not test_run:
                    plt.show()

                plt.close("all")
                plt.figure(num=1, dpi=240)
                plt.clf()
                ax = plt.gca()
                gncs_draw_curve(plt, av_time_gnc_welsch_list,    ("SupGN",     "Welsch",      "GNC_Welsch" ), xvalues=outlier_fraction_list)
                gncs_draw_curve(plt, av_time_huber_list,         ("SupGN",     "PseudoHuber", "PseudoHuber"), xvalues=outlier_fraction_list)
                gncs_draw_curve(plt, av_time_gnc_irls_p_list,    ("SupGN",     "GNC_IRLSp",   "GNC_IRLSp0" ), xvalues=outlier_fraction_list)

                ax.set_xlabel(r'Outlier fraction' )
                ax.set_ylabel(r'Average time in msec')
                ax.set_yscale("log")
                #plt.box(False)
                ax.set_xlim(0.0,outlier_fraction_list[len(outlier_fraction_list)-1])

                plt.legend()
                plt.savefig(os.path.join(output_folder, "av_time_known_scale_n" + str(n) + "_range" + str(int(x_range)) + ".png"), bbox_inches='tight')
                if False: #not test_run:
                    plt.show()

            plt.close("all")
            plt.figure(num=1, dpi=240)
            plt.clf()
            ax = plt.gca()
            gncs_draw_curve(plt, eff_theil_sen_list,      ("Theil-Sen",     "",               ""         ), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, eff_ransac_list,         ("RANSAC",        "",               ""         ), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, eff_hough_list,          ("Hough",         "",               ""         ), xvalues=outlier_fraction_list)
            #gncs_draw_curve(plt, eff_ls_list,             ("LS",            "",               ""         ), xvalues=outlier_fraction_list)

            ax.set_xlabel(r'Outlier fraction' )
            ax.set_ylabel('Relative efficiency')
            #plt.box(False)
            ax.set_xlim(0.0,outlier_fraction_list[len(outlier_fraction_list)-1])
            ax.set_ylim(0.0,1.1)

            plt.legend()
            plt.savefig(os.path.join(output_folder, "efficiency_n" + str(n) + "_range" + str(int(x_range)) + ".png"), bbox_inches='tight')
            if False: #not test_run:
                plt.show()

    if test_run:
        print("line_fit_efficiency OK")

if __name__ == "__main__":
    main(False, quick_run=True) # test_run
