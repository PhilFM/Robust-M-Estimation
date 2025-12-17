import numpy as np
import matplotlib.pyplot as plt
import json
import os

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

from gnc_smoothie_philfm.plt_alg_vis import gncs_draw_curve

from mean_compare_apply import mean_compare_apply

def main(test_run:bool, output_folder:str="../../output", quick_run:bool=False):
    sigma_pop = 1.0
    p = 0.66666667

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
            eff_gncirlsp_list = []
            eff_rme_list = []
            data_dict = {}
            for outlier_fraction in outlier_fraction_list:
                output_file = '' # '../../output/solver-' + str(int(xgtrange)) + "-" + str(n) + "-" + str(int(100.0*outlier_fraction)) + ".png"
                data_array,mgt,sdgncwelsch,sdmean,sdhuber,sdtrimmed,sdmedian,sdgncirlsp,sdrme,n_samples = mean_compare_apply(sigma_pop, p, xgtrange, n, n_samples_base, min_n_samples, outlier_fraction, output_file=output_file, test_run=test_run)

                outlier_dict = {}
                outlier_dict['data'] = data_array
                outlier_dict['popmean'] = mgt
                efficiency_dict = {}

                eff = sigma_pop*sigma_pop/(n*sdgncwelsch*sdgncwelsch)
                if not test_run:
                    print("GNC Welsch estimator efficiency: ", eff)

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
                    print("GNC IRLS-p=0 estimator efficiency: ", eff)

                eff_gncirlsp_list.append(eff)
                efficiency_dict['GNC IRLS-p'] = eff

                eff = sigma_pop*sigma_pop/(n*sdrme*sdrme)
                if not test_run:
                    print("Robust Mean Estimator efficiency: ", eff)

                eff_rme_list.append(eff)
                efficiency_dict['RME'] = eff

                outlier_dict['efficiency'] = efficiency_dict
                data_dict['outlier_fraction-'+str(outlier_fraction)] = outlier_dict

            data_dict['n_samples'] = n_samples
            jstr = json.dumps(data_dict)
            js = json.loads(jstr)
            with open(os.path.join(output_folder, "compare_n" + str(n) + "_range" + str(int(xgtrange)) + ".json"), 'w', encoding='utf-8') as f:
                json.dump(js, f, ensure_ascii=False, indent=4)

            plt.close("all")
            plt.figure(num=1, dpi=240)
            plt.clf()
            ax = plt.gca()
            gncs_draw_curve(plt, eff_gncwelsch_list,    ("IRLS",   "Welsch",      "GNC_Welsch"), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, eff_mean_list,         ("Mean",   "Basic",       ""          ), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, eff_huber_list,        ("IRLS",   "PseudoHuber", "Welsch"    ), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, eff_trimmed_list,      ("Mean",   "Trimmed",     ""          ), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, eff_median_list,       ("Median", "Basic",       ""          ), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, eff_gncirlsp_list,     ("IRLS",   "GNC_IRLSp",   "GNC_IRLSp0"), xvalues=outlier_fraction_list)
            gncs_draw_curve(plt, eff_rme_list,          ("RME",    "",            ""          ), xvalues=outlier_fraction_list)

            ax.set_xlabel(r'Outlier fraction' )
            ax.set_ylabel('Relative efficiency')
            #plt.box(False)
            ax.set_xlim(0.0,outlier_fraction_list[len(outlier_fraction_list)-1])
            ax.set_ylim(0.0,1.1)

            plt.legend()
            plt.savefig(os.path.join(output_folder, "compare_n" + str(n) + "_range" + str(int(xgtrange)) + ".png"), bbox_inches='tight')
            if not test_run:
                plt.show()

    if test_run:
        print("mean_compare OK")

if __name__ == "__main__":
    main(True) # test_run
