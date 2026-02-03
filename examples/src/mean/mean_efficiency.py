import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

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
    n_samples_base = 1000 if quick_run else 50000
    min_n_samples = 40 if quick_run else 1000

    np.random.seed(0) # We want the numbers to be the same on each run

    outlier_fraction_list = [0.0,0.2,0.4] if quick_run else [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for xgtrange in [3.0]: #,5.0,10.0,30.0,100.0]:
        sample_size_array = [10] if quick_run else [10,30,100,1000]
        for n in sample_size_array:
            eff_gncwelsch_list = []
            eff_trimmed_list = []
            eff_median_list = []
            eff_trimean_list = []
            for outlier_fraction in outlier_fraction_list:
                output_file_1 = None # Path("../../../output/efficiency1-" + str(int(xgtrange)) + "-" + str(n) + "-" + str(int(100.0*outlier_fraction)) + ".png")
                output_file_2 = None # Path("../../../output/efficiency2-" + str(int(xgtrange)) + "-" + str(n) + "-" + str(int(100.0*outlier_fraction)) + ".png")
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
                    output_folder=output_folder,
                    smoothie=True)

                eff = sigma_pop*sigma_pop/(n*alg_result.sd_gnc_welsch*alg_result.sd_gnc_welsch)
                if not test_run:
                    print("GNC Welsch estimator efficiency: ", eff)

                eff_gncwelsch_list.append(eff)

                eff = sigma_pop*sigma_pop/(n*alg_result.sd_trimmed*alg_result.sd_trimmed)
                if not test_run:
                    print("Trimmed mean 50% efficiency: ", eff)

                eff_trimmed_list.append(eff)

                eff = sigma_pop*sigma_pop/(n*alg_result.sd_median*alg_result.sd_median)
                if not test_run:
                    print("Median efficiency: ", eff)

                eff_median_list.append(eff)

                eff = sigma_pop*sigma_pop/(n*alg_result.sd_trimean*alg_result.sd_trimean)
                if not test_run:
                    print("Tukey trimean efficiency: ", eff)

                eff_trimean_list.append(eff)

            plt.close("all")
            plt.figure(num=1, dpi=240)
            plt.clf()
            ax = plt.gca()
            gncs_draw_curve(plt, eff_gncwelsch_list,    ("SupGN",  "Welsch",      "GNC_Welsch"), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, eff_trimmed_list,      ("Mean",   "Trimmed",     ""          ), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, eff_median_list,       ("Median", "Basic",       ""          ), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, eff_trimean_list,      ("Trimean","Basic",       ""          ), xvalues=outlier_fraction_list)

            ax.set_xlabel(r'Outlier fraction' )
            ax.set_ylabel('Relative efficiency')
            #plt.box(False)
            ax.set_xlim(0.0,outlier_fraction_list[len(outlier_fraction_list)-1])
            ax.set_ylim(0.0,1.1)

            plt.legend()
            plt.savefig(os.path.join(output_folder, "mean_efficiency_n" + str(n) + "_range" + str(int(xgtrange)) + ".png"), bbox_inches='tight')
            if not test_run:
                plt.show()

    if test_run:
        print("mean_efficiency OK")

if __name__ == "__main__":
    main(True, quick_run=False) # test_run
