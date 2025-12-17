import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import json
import os

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

from gnc_smoothie_philfm.plt_alg_vis import gncs_draw_curve

from mean_compare_apply import mean_compare_apply

def main(test_run:bool, output_folder:str="../../output", quick_run:bool=False):
    sigma_pop = 1.0
    p = 0.5 # 33333333

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
        eff_gncirlsp_list = []
        eff_rme_list = []
        data_dict = {}
        for student_t_dof in student_t_dof_list:
            output_file = '' #'../../output/test.png' if student_t_dof == 2 else ''
            data_array,mgt,sdgncwelsch,sdmean,sdhuber,sdtrimmed,sdmedian,sdgncirlsp,sdrme,n_samples = mean_compare_apply(sigma_pop, p, xgtrange, n, n_samples_base, min_n_samples, 0.0, student_t_dof=student_t_dof, output_file=output_file, test_run=test_run)

            outlier_dict = {}
            outlier_dict['data'] = data_array
            outlier_dict['popmean'] = mgt
            efficiency_dict = {}

            eff = sigma_pop*sigma_pop/(n*sdgncwelsch*sdgncwelsch)
            if not test_run:
                print("SUP-GN GNC-Welsch estimator efficiency: ", eff)

            eff_gncwelsch_list.append(eff)
            efficiency_dict['GNC Welsch'] = eff

            eff = sigma_pop*sigma_pop/(n*sdmean*sdmean)
            if not test_run:
                print("Mean efficiency: ", eff)

            eff_mean_list.append(eff)
            efficiency_dict['mean'] = eff

            eff = sigma_pop*sigma_pop/(n*sdhuber*sdhuber)
            if not test_run:
                print("Pseudo-Huber estimator efficiency: ", eff)

            eff_huber_list.append(eff)
            efficiency_dict['Pseudo-Huber'] = eff

            eff = sigma_pop*sigma_pop/(n*sdtrimmed*sdtrimmed)
            if not test_run:
                print("Trimmed mean 50% efficiency: ", eff)

            eff_trimmed_list.append(eff)
            efficiency_dict['trimmed mean 50%'] = eff

            eff = sigma_pop*sigma_pop/(n*sdmedian*sdmedian)
            if not test_run:
                print("Median efficiency: ", eff)

            eff_median_list.append(eff)
            efficiency_dict['median'] = eff

            eff = sigma_pop*sigma_pop/(n*sdgncirlsp*sdgncirlsp)
            if not test_run:
                print("GNC IRLS-p estimator efficiency: ", eff)

            eff_gncirlsp_list.append(eff)
            efficiency_dict['GNC IRLS-p'] = eff

            eff = sigma_pop*sigma_pop/(n*sdrme*sdrme)
            if not test_run:
                print("Robust Mean Estimator efficiency: ", eff)

            eff_rme_list.append(eff)
            efficiency_dict['RME'] = eff

            outlier_dict['efficiency'] = efficiency_dict
            data_dict['DOF-'+str(student_t_dof)] = outlier_dict

        data_dict['n_samples'] = n_samples
        jstr = json.dumps(data_dict)
        js = json.loads(jstr)
        with open(os.path.join(output_folder, "compare_student_t_n" + str(n) + ".json"), 'w', encoding='utf-8') as f:
            json.dump(js, f, ensure_ascii=False, indent=4)

        plt.close("all")
        plt.figure(num=1, dpi=240)
        plt.clf()
        ax = plt.gca()
        gncs_draw_curve(plt, eff_gncwelsch_list,    ("IRLS",   "Welsch",      "GNC_Welsch"), xvalues=student_t_dof_list)
        gncs_draw_curve(plt, eff_mean_list,         ("Mean",   "Basic",       ""          ), xvalues=student_t_dof_list)
        gncs_draw_curve(plt, eff_huber_list,        ("IRLS",   "PseudoHuber", "Welsch"    ), xvalues=student_t_dof_list)
        gncs_draw_curve(plt, eff_trimmed_list,      ("Mean",   "Trimmed",     ""          ), xvalues=student_t_dof_list)
        gncs_draw_curve(plt, eff_median_list,       ("Median", "Basic",       ""          ), xvalues=student_t_dof_list)
        gncs_draw_curve(plt, eff_gncirlsp_list,     ("IRLS",   "GNC_IRLSp",   "GNC_IRLSp0"), xvalues=student_t_dof_list)
        gncs_draw_curve(plt, eff_rme_list,          ("RME",    "",            ""          ), xvalues=student_t_dof_list)

        ax.set_xlabel(r'Degrees of freedom' )
        ax.set_ylabel('Relative efficiency')
        #plt.box(False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(student_t_dof_list[0],student_t_dof_list[len(student_t_dof_list)-1])
        ax.set_ylim(0.0,1.1)

        plt.legend()
        plt.savefig(os.path.join(output_folder, "compare_student_t_n" + str(n) + ".png"), bbox_inches='tight')
        if not test_run:
            plt.show()

    if test_run:
        print("mean_compare_student_t OK")

if __name__ == "__main__":
    main(False) # test_run
