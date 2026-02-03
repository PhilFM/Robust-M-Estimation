import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
#from pathlib import Path
import os

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

    # data range
    xgtrange = 50.0

    # number of samples used for statistics
    n_samples_base = 200 if quick_run else 50000
    min_n_samples = 4 if quick_run else 1000

    np.random.seed(0) # We want the numbers to be the same on each run

    student_t_dof_list = [1,2,3,4,5]
    sample_size_array = [10,20] if quick_run else [10,20,50,100]
    for n in sample_size_array:
        eff_gncwelsch_list = []
        eff_mean_list = []
        eff_huber_list = []
        eff_trimmed_list = []
        eff_median_list = []
        eff_trimean_list = []
        eff_gncirlsp_list = []
        eff_rme_list = []
        for student_t_dof in student_t_dof_list:
            output_file_1 = None #Path("../../../output/test1.png") if student_t_dof == 2 else None
            output_file_2 = None #Path("../../../output/test2.png") if student_t_dof == 2 else None
            alg_result = mean_compare_apply(
                sigma_pop,
                xgtrange,
                n,
                n_samples_base,
                min_n_samples,
                0.0,
                welsch_q,
                pseudo_huber_sigma_scale,
                gnc_irls_p_epsilon_scale,
                rme_beta_scale,
                student_t_dof=student_t_dof,
                output_file_1=output_file_1,
                output_file_2=output_file_2,
                test_run=test_run,
                output_folder=output_folder)

            eff = sigma_pop*sigma_pop/(n*alg_result.sd_gnc_welsch*alg_result.sd_gnc_welsch)
            if not test_run:
                print("SUP-GN GNC-Welsch estimator efficiency: ", eff)

            eff_gncwelsch_list.append(eff)

            eff = sigma_pop*sigma_pop/(n*alg_result.sd_mean*alg_result.sd_mean)
            if not test_run:
                print("Mean efficiency: ", eff)

            eff_mean_list.append(eff)

            eff = sigma_pop*sigma_pop/(n*alg_result.sd_huber*alg_result.sd_huber)
            if not test_run:
                print("Pseudo-Huber estimator efficiency: ", eff)

            eff_huber_list.append(eff)

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
                print("Trimean efficiency: ", eff)

            eff_trimean_list.append(eff)

            eff = sigma_pop*sigma_pop/(n*alg_result.sd_gnc_irls_p*alg_result.sd_gnc_irls_p)
            if not test_run:
                print("GNC IRLS-p estimator efficiency: ", eff)

            eff_gncirlsp_list.append(eff)

            eff = sigma_pop*sigma_pop/(n*alg_result.sd_rme*alg_result.sd_rme)
            if not test_run:
                print("Robust Mean Estimator efficiency: ", eff)

            eff_rme_list.append(eff)

        plt.close("all")
        plt.figure(num=1, dpi=240)
        plt.clf()
        ax = plt.gca()
        gncs_draw_curve(plt, eff_gncwelsch_list,    ("IRLS",   "Welsch",      "GNC_Welsch"), xvalues=student_t_dof_list)
        gncs_draw_curve(plt, eff_mean_list,         ("Mean",   "Basic",       ""          ), xvalues=student_t_dof_list)
        gncs_draw_curve(plt, eff_trimmed_list,      ("Mean",   "Trimmed",     ""          ), xvalues=student_t_dof_list)
        gncs_draw_curve(plt, eff_median_list,       ("Median", "Basic",       ""          ), xvalues=student_t_dof_list)
        gncs_draw_curve(plt, eff_trimean_list,      ("Trimean","Basic",       ""          ), xvalues=student_t_dof_list)

        ax.set_xlabel(r'Degrees of freedom' )
        ax.set_ylabel('Relative efficiency')
        #plt.box(False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(student_t_dof_list[0],student_t_dof_list[len(student_t_dof_list)-1])
        ax.set_ylim(0.0,1.1)

        plt.legend()
        plt.savefig(os.path.join(output_folder, "compare_student_t_n" + str(n) + "_lref.png"), bbox_inches='tight')
        if not test_run:
            plt.show()

        plt.close("all")
        plt.figure(num=1, dpi=240)
        plt.clf()
        ax = plt.gca()
        gncs_draw_curve(plt, eff_gncwelsch_list,    ("IRLS",   "Welsch",      "GNC_Welsch"), xvalues=student_t_dof_list)
        gncs_draw_curve(plt, eff_huber_list,        ("IRLS",   "PseudoHuber", "Welsch"    ), xvalues=student_t_dof_list)
        gncs_draw_curve(plt, eff_gncirlsp_list,     ("IRLS",   "GNC_IRLSp",   "GNC_IRLSp0"), xvalues=student_t_dof_list)
        gncs_draw_curve(plt, eff_rme_list,          ("RME",    "",            ""          ), xvalues=student_t_dof_list)

        ax.set_xlabel(r'Degrees of freedom' )
        ax.set_ylabel('Relative efficiency')
        #plt.box(False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(student_t_dof_list[0],student_t_dof_list[len(student_t_dof_list)-1])
        ax.set_ylim(0.0,1.1)

        plt.legend()
        plt.savefig(os.path.join(output_folder, "compare_student_t_n" + str(n) + "_mref.png"), bbox_inches='tight')
        if not test_run:
            plt.show()

    if test_run:
        print("mean_compare_student_t OK")

if __name__ == "__main__":
    main(True) # test_run
