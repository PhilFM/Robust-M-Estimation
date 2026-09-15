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
    output_folder += "/regression/complexity"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    sigma_pop = 0.1
    welsch_q = 0.66666667
    huber_sigma_scale = 0.6
    gnc_irls_p_epsilon_scale = 1.4
    outlier_fraction = 0.02
    x_range = 10.0

    # number of samples used for statistics
    n_samples_base = 100 if quick_run else 100
    min_n_samples = 10 if quick_run else 10

    np.random.seed(0) # We want the numbers to be the same on each run

    n_dimensions_list = [2,5] if quick_run else range(2,46,4)
    sample_size_array = [300] if quick_run else [100,300,1000]

    if test_idx >= 0:
        tidx = -1

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
        eff_ls_list = []

        av_time_gnc_welsch_list = []
        av_time_huber_list = []
        av_time_gnc_irls_p_list = []
        av_time_mm_estimation_list = []
        av_time_theil_sen_list = []
        av_time_ransac_list = []
        av_time_ls_list = []
        for ndim in n_dimensions_list:
            alg_result = apply_to_data(ndim, sigma_pop, welsch_q, huber_sigma_scale, gnc_irls_p_epsilon_scale, x_range, n_data_points, n_samples_base, min_n_samples, outlier_fraction, test_run=test_run)

            # Huber
            if False:
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

            if False:
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
                print("   GNC IRLS-p average time", 1000.0*alg_result.av_time_gnc_irls_p, "msec")

            # GNC IRLS Welsch
            if False:
                fac = np.linalg.cholesky(alg_result.var_gnc_welsch)
                facI = np.linalg.inv(fac)
                eff_mat = np.matmul(facI, np.matmul(alg_result.var_ls, np.matrix.transpose(facI)))
                eff = math.pow(np.linalg.det(eff_mat),1.0/ndim)
                if not test_run:
                    print("   GNC Welsch estimator SS efficiency: ", eff)

                eff_gnc_welsch_list.append(eff)

            av_time_gnc_welsch_list.append(1000.0*alg_result.av_time_gnc_welsch)
            if not test_run:
                print("   GNC Welsch estimator SS average time", 1000.0*alg_result.av_time_gnc_welsch, "msec")

            # MM estimation
            if False:
                fac = np.linalg.cholesky(alg_result.var_mm_estimation)
                facI = np.linalg.inv(fac)
                eff_mat = np.matmul(facI, np.matmul(alg_result.var_ls, np.matrix.transpose(facI)))
                eff = math.pow(np.linalg.det(eff_mat),1.0/ndim)
                if not test_run:
                    print("   MM estimation efficiency: ", eff, "average time")

                eff_mm_estimation_list.append(eff)

            av_time_mm_estimation_list.append(1000.0*alg_result.av_time_mm_estimation)
            if not test_run:
                print("   MM estimation average time", 1000.0*alg_result.av_time_mm_estimation, "msec")

            # Theil-Sen
            if False:
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
            if False:
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

        n_values = len(n_dimensions_list)
        if False:
            plt.close("all")
            plt.figure(num=1, dpi=240)
            plt.clf()
            ax = plt.gca()
            gncs_draw_curve(plt, eff_gnc_welsch_list,    ("SupGN",     "Welsch",      "SS"         ), xvalues=n_dimensions_list)
            gncs_draw_curve(plt, eff_huber_list,         ("SupGN",     "PseudoHuber", "PseudoHuber"), xvalues=n_dimensions_list)
            gncs_draw_curve(plt, eff_gnc_irls_p_list,    ("SupGN",     "GNC_IRLSp",   "GNC_IRLSp0" ), xvalues=n_dimensions_list)

            ax.set_xlabel(r'Dimension' )
            ax.set_ylabel('Relative efficiency')
            ax.set_xlim(0.0,n_dimensions_list[n_values-1])
            ax.set_ylim(0.0,1.1)

            plt.legend()
            plt.savefig(os.path.join(output_folder, "dim_efficiency_known_scale_n" + str(n_data_points) + ".png"), bbox_inches='tight')
            if False: #not test_run:
                plt.show()

        plt.close("all")
        plt.figure(num=1, dpi=240)
        plt.clf()
        ax = plt.gca()
        gncs_draw_curve(plt, av_time_gnc_welsch_list,    ("SupGN",     "Welsch",      "SS"         ), xvalues=n_dimensions_list)
        gncs_draw_curve(plt, av_time_huber_list,         ("SupGN",     "PseudoHuber", "PseudoHuber"), xvalues=n_dimensions_list)
        gncs_draw_curve(plt, av_time_gnc_irls_p_list,    ("SupGN",     "GNC_IRLSp",   "GNC_IRLSp0" ), xvalues=n_dimensions_list)

        ax.set_xlabel(r'Dimension' )
        ax.set_ylabel(r'Average time in msec')
        ax.set_yscale("log")
        #plt.box(False)
        ax.set_xlim(0.0,n_dimensions_list[n_values-1])

        plt.legend()
        plt.savefig(os.path.join(output_folder, "dim_av_time_known_scale_n" + str(n_data_points) + ".png"), bbox_inches='tight')
        if False: #not test_run:
            plt.show()

        if False:
            plt.close("all")
            plt.figure(num=1, dpi=240)
            plt.clf()
            ax = plt.gca()
            gncs_draw_curve(plt, eff_gnc_welsch_list,     ("SupGN",         "Welsch",         "SS"       ), xvalues=n_dimensions_list)
            gncs_draw_curve(plt, eff_mm_estimation_list,  ("MM-Estimation", "Tukey-Bisquare", "Large-Rho"), xvalues=n_dimensions_list)
            gncs_draw_curve(plt, eff_theil_sen_list,      ("Theil-Sen",     "",               ""         ), xvalues=n_dimensions_list)
            gncs_draw_curve(plt, eff_ransac_list,         ("RANSAC",        "",               ""         ), xvalues=n_dimensions_list)

            ax.set_xlabel(r'Dimension' )
            ax.set_ylabel('Relative efficiency')
            #plt.box(False)
            ax.set_xlim(0.0,n_dimensions_list[n_values-1])
            ax.set_ylim(0.0,1.1)

            plt.legend()
            plt.savefig(os.path.join(output_folder, "dim_efficiency_n" + str(n_data_points) + ".png"), bbox_inches='tight')
            if False: #not test_run:
                plt.show()

        plt.close("all")
        plt.figure(num=1, dpi=240)
        plt.clf()
        ax = plt.gca()
        gncs_draw_curve(plt, av_time_gnc_welsch_list,     ("SupGN",         "Welsch",         "SS"       ), xvalues=n_dimensions_list)
        gncs_draw_curve(plt, av_time_mm_estimation_list,  ("MM-Estimation", "Tukey-Bisquare", "Large-Rho"), xvalues=n_dimensions_list)
        gncs_draw_curve(plt, av_time_theil_sen_list,      ("Theil-Sen",     "",               ""         ), xvalues=n_dimensions_list)
        gncs_draw_curve(plt, av_time_ransac_list,         ("RANSAC",        "",               ""         ), xvalues=n_dimensions_list)
        #gncs_draw_curve(plt, av_time_ls_list,             ("LS",            "",               ""         ), xvalues=n_dimensions_list)

        ax.set_xlabel(r'Dimension' )
        ax.set_ylabel(r'Average time in msec')
        ax.set_yscale("log")
        #plt.box(False)
        #ax.set_xlim(0.0,n_dimensions_list[n_values-1])
        #ax.set_ylim(0.0,1.1)

        plt.legend()
        plt.savefig(os.path.join(output_folder, "dim_av_time_n" + str(n_data_points) + ".png"), bbox_inches='tight')
        if False: #not test_run:
            plt.show()

        plt.close("all")
        plt.figure(num=1, dpi=240)
        plt.clf()
        ax = plt.gca()
        gncs_draw_curve(plt, av_time_gnc_welsch_list,     ("SupGN",         "Welsch",         "SS"       ), xvalues=n_dimensions_list)

        ax.set_xlabel(r'Dimension' )
        ax.set_ylabel(r'Average time in msec')
        #plt.box(False)
        #ax.set_xlim(0.0,n_dimensions_list[n_values-1])
        ax.set_ylim(0.0,1.05*max(av_time_gnc_welsch_list))

        plt.legend()
        plt.savefig(os.path.join(output_folder, "dim_av_time_gnc_welsch_n" + str(n_data_points) + ".png"), bbox_inches='tight')
        if False: #not test_run:
            plt.show()

        plt.close("all")
        plt.figure(num=1, dpi=240)
        plt.clf()
        ax = plt.gca()
        gncs_draw_curve(plt, av_time_mm_estimation_list,  ("MM-Estimation", "Tukey-Bisquare", "Large-Rho"), xvalues=n_dimensions_list)

        ax.set_xlabel(r'Dimension' )
        ax.set_ylabel(r'Average time in msec')
        #plt.box(False)
        #ax.set_xlim(0.0,n_dimensions_list[n_values-1])
        ax.set_ylim(0.0,1.05*max(av_time_mm_estimation_list))

        plt.legend()
        plt.savefig(os.path.join(output_folder, "dim_av_time_mm_estimation_n" + str(n_data_points) + ".png"), bbox_inches='tight')
        if False: #not test_run:
            plt.show()

        plt.close("all")
        plt.figure(num=1, dpi=240)
        plt.clf()
        ax = plt.gca()
        gncs_draw_curve(plt, av_time_theil_sen_list,      ("Theil-Sen",     "",               ""         ), xvalues=n_dimensions_list)

        ax.set_xlabel(r'Dimension' )
        ax.set_ylabel(r'Average time in msec')
        #plt.box(False)
        #ax.set_xlim(0.0,n_dimensions_list[n_values-1])
        ax.set_ylim(0.0,1.05*max(av_time_theil_sen_list))

        plt.legend()
        plt.savefig(os.path.join(output_folder, "dim_av_time_theil_sen_n" + str(n_data_points) + ".png"), bbox_inches='tight')
        if False: #not test_run:
            plt.show()

        plt.close("all")
        plt.figure(num=1, dpi=240)
        plt.clf()
        ax = plt.gca()
        gncs_draw_curve(plt, av_time_ransac_list,         ("RANSAC",        "",               ""         ), xvalues=n_dimensions_list)

        ax.set_xlabel(r'Dimension' )
        ax.set_ylabel(r'Average time in msec')
        #plt.box(False)
        #ax.set_xlim(0.0,n_dimensions_list[n_values-1])
        ax.set_ylim(0.0,1.05*max(av_time_ransac_list))

        plt.legend()
        plt.savefig(os.path.join(output_folder, "dim_av_time_ransac_n" + str(n_data_points) + ".png"), bbox_inches='tight')
        if False: #not test_run:
            plt.show()

    if test_run:
        print("regression_complexity_dimension OK")

if __name__ == "__main__":
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-tidx", "--testindex", type=int, required=False, help="test index", default = -1)
    args = vars(ap.parse_args())
    #print("args=",args)
    #print("test_index=",args["testindex"])
    
    main(test_idx=args["testindex"], test_run=False, quick_run=False) # test_run
