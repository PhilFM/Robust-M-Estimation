import numpy as np
import matplotlib.pyplot as plt
import os
#from pathlib import Path

if __name__ == "__main__":
    import sys
    sys.path.append("../../../pypi_package/src")

from gnc_smoothie_philfm.plt_alg_vis import gncs_draw_curve

from mean_compare_apply import mean_compare_apply

def main(test_run:bool, output_folder:str="../../../output", quick_run:bool=False):
    sigma_pop = 1.0
    welsch_q = 0.62
    pseudo_huber_sigma_scale = 0.6
    gnc_irls_p_epsilon_scale = 1.4
    rme_beta_scale = 0.95

    # number of samples used for statistics
    n_samples_base = 100 if quick_run else 50000
    min_n_samples = 4 if quick_run else 1000

    np.random.seed(0) # We want the numbers to be the same on each run

    outlier_fraction_list = [0.0,0.2,0.5] if quick_run else [0.0,0.1,0.2,0.3,0.4,0.5]
    for xgtrange in [3.0,5.0,10.0,30.0,100.0]:
        sample_size_array = [10] if quick_run else [10,30,100,1000]
        for n in sample_size_array:
            eff_gncwelsch_list = []
            eff_mean_list = []
            eff_huber_list = []
            eff_trimmed_list = []
            eff_median_list = []
            eff_trimean_list = []
            eff_gncirlsp_list = []
            eff_rme_list = []

            time_gncwelsch_list = []
            time_mean_list = []
            time_huber_list = []
            time_trimmed_list = []
            time_median_list = []
            time_trimean_list = []
            time_gncirlsp_list = []
            time_rme_list = []
            for outlier_fraction in outlier_fraction_list:
                output_file_1 = None # Path("../../../output/solver1-" + str(int(xgtrange)) + "-" + str(n) + "-" + str(int(100.0*outlier_fraction)) + ".png")
                output_file_2 = None # Path("../../../output/solver2-" + str(int(xgtrange)) + "-" + str(n) + "-" + str(int(100.0*outlier_fraction)) + ".png")
                alg_result = mean_compare_apply(
                    sigma_pop,
                    xgtrange,
                    n,
                    n_samples_base,
                    min_n_samples,
                    outlier_fraction,
                    welsch_q,
                    pseudo_huber_sigma_scale,
                    gnc_irls_p_epsilon_scale,
                    rme_beta_scale,
                    output_file_1=output_file_1,
                    output_file_2=output_file_2,
                    test_run=test_run,
                    output_folder=output_folder)

                eff = sigma_pop*sigma_pop/(n*alg_result.sd_gnc_welsch*alg_result.sd_gnc_welsch)
                if not test_run:
                    print("GNC Welsch estimator efficiency: ", eff)

                eff_gncwelsch_list.append(eff)
                time_gncwelsch_list.append(alg_result.time_gnc_welsch)

                eff = sigma_pop*sigma_pop/(n*alg_result.sd_mean*alg_result.sd_mean)
                if not test_run:
                    print("Mean efficiency: ", eff)

                eff_mean_list.append(eff)
                time_mean_list.append(alg_result.time_mean)

                eff = sigma_pop*sigma_pop/(n*alg_result.sd_huber*alg_result.sd_huber)
                if not test_run:
                    print("Pseudo-Huber estimator efficiency: ", eff)

                eff_huber_list.append(eff)
                time_huber_list.append(alg_result.time_huber)

                eff = sigma_pop*sigma_pop/(n*alg_result.sd_trimmed*alg_result.sd_trimmed)
                if not test_run:
                    print("Trimmed mean 50% efficiency: ", eff)

                eff_trimmed_list.append(eff)
                time_trimmed_list.append(alg_result.time_trimmed)

                eff = sigma_pop*sigma_pop/(n*alg_result.sd_median*alg_result.sd_median)
                if not test_run:
                    print("Median efficiency: ", eff)

                eff_median_list.append(eff)
                time_median_list.append(alg_result.time_median)

                eff = sigma_pop*sigma_pop/(n*alg_result.sd_trimean*alg_result.sd_trimean)
                if not test_run:
                    print("Tukey trimean efficiency: ", eff)

                eff_trimean_list.append(eff)
                time_trimean_list.append(alg_result.time_trimean)

                eff = sigma_pop*sigma_pop/(n*alg_result.sd_gnc_irls_p*alg_result.sd_gnc_irls_p)
                if not test_run:
                    print("GNC IRLS-p=0 estimator efficiency: ", eff)

                eff_gncirlsp_list.append(eff)
                time_gncirlsp_list.append(alg_result.time_gnc_irls_p)

                eff = sigma_pop*sigma_pop/(n*alg_result.sd_rme*alg_result.sd_rme)
                if not test_run:
                    print("Robust Mean Estimator efficiency: ", eff)

                eff_rme_list.append(eff)
                time_rme_list.append(alg_result.time_rme)

            plt.close("all")
            plt.figure(num=1, dpi=240)
            plt.clf()
            ax = plt.gca()
            gncs_draw_curve(plt, eff_gncwelsch_list,    ("IRLS",   "Welsch",      "GNC_Welsch"), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, eff_mean_list,         ("Mean",   "Basic",       ""          ), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, eff_trimmed_list,      ("Mean",   "Trimmed",     ""          ), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, eff_median_list,       ("Median", "Basic",       ""          ), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, eff_trimean_list,      ("Trimean","Basic",       ""          ), xvalues=outlier_fraction_list)

            ax.set_xlabel(r'Outlier fraction' )
            ax.set_ylabel('Relative efficiency')
            #plt.box(False)
            ax.set_xlim(0.0,outlier_fraction_list[len(outlier_fraction_list)-1])
            ax.set_ylim(0.0,1.1)

            plt.legend()
            plt.savefig(os.path.join(output_folder, "compare_n" + str(n) + "_range" + str(int(xgtrange)) + "_lref.png"), bbox_inches='tight')
            if False: #not test_run:
                plt.show()

            plt.close("all")
            plt.figure(num=1, dpi=240)
            plt.clf()
            ax = plt.gca()
            gncs_draw_curve(plt, time_gncwelsch_list,    ("IRLS",   "Welsch",      "GNC_Welsch"), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, time_mean_list,         ("Mean",   "Basic",       ""          ), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, time_trimmed_list,      ("Mean",   "Trimmed",     ""          ), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, time_median_list,       ("Median", "Basic",       ""          ), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, time_trimean_list,      ("Trimean","Basic",       ""          ), xvalues=outlier_fraction_list)

            ax.set_xlabel(r'Outlier fraction' )
            ax.set_ylabel('Execution time')
            #plt.box(False)
            ax.set_xlim(0.0,outlier_fraction_list[len(outlier_fraction_list)-1])
            #ax.set_ylim(0.0,1.1)

            plt.legend()
            plt.savefig(os.path.join(output_folder, "compare_time1_n" + str(n) + "_range" + str(int(xgtrange)) + ".png"), bbox_inches='tight')
            if False: #not test_run:
                plt.show()

            plt.close("all")
            plt.figure(num=1, dpi=240)
            plt.clf()
            ax = plt.gca()
            gncs_draw_curve(plt, eff_gncwelsch_list,    ("IRLS",   "Welsch",      "GNC_Welsch"), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, eff_huber_list,        ("IRLS",   "PseudoHuber", "Welsch"    ), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, eff_gncirlsp_list,     ("IRLS",   "GNC_IRLSp",   "GNC_IRLSp0"), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, eff_rme_list,          ("RME",    "",            ""          ), xvalues=outlier_fraction_list)

            ax.set_xlabel(r'Outlier fraction' )
            ax.set_ylabel('Relative efficiency')
            #plt.box(False)
            ax.set_xlim(0.0,outlier_fraction_list[len(outlier_fraction_list)-1])
            ax.set_ylim(0.0,1.1)

            plt.legend()
            plt.savefig(os.path.join(output_folder, "compare_n" + str(n) + "_range" + str(int(xgtrange)) + "_mref.png"), bbox_inches='tight')
            if False: #not test_run:
                plt.show()

            plt.close("all")
            plt.figure(num=1, dpi=240)
            plt.clf()
            ax = plt.gca()
            gncs_draw_curve(plt, time_gncwelsch_list,    ("IRLS",   "Welsch",      "GNC_Welsch"), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, time_huber_list,        ("IRLS",   "PseudoHuber", "Welsch"    ), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, time_gncirlsp_list,     ("IRLS",   "GNC_IRLSp",   "GNC_IRLSp0"), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, time_rme_list,          ("RME",    "",            ""          ), xvalues=outlier_fraction_list)

            ax.set_xlabel(r'Outlier fraction' )
            ax.set_ylabel('Execution time')
            #plt.box(False)
            ax.set_xlim(0.0,outlier_fraction_list[len(outlier_fraction_list)-1])
            #ax.set_ylim(0.0,1.1)

            plt.legend()
            plt.savefig(os.path.join(output_folder, "compare_time2_n" + str(n) + "_range" + str(int(xgtrange)) + ".png"), bbox_inches='tight')
            if False: #not test_run:
                plt.show()

    if test_run:
        print("mean_compare OK")

if __name__ == "__main__":
    main(False) # test_run
