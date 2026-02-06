import numpy as np
import matplotlib.pyplot as plt
import os
import math

if __name__ == "__main__":
    import sys
    sys.path.append("../../../pypi_package/src")

from gnc_smoothie.plt_alg_vis import gncs_draw_curve

from trs_welsch import TRSWelsch
from trs import TRS

def fit_trs_ls(data):
    trs = TRS()
    S = np.zeros((4,4))
    a = np.zeros(4)
    for d in data:
        grad = trs.residual_gradient(d)
        gradT = np.matrix.transpose(grad)
        S += np.matmul(gradT,grad)
        p = np.matmul(gradT,d[2:4])
        a += p

    Si = np.linalg.inv(S)
    return -np.matmul(Si, a)

def randomM11() -> float:
    return 2.0*(np.random.rand()-0.5)

def apply_to_data(sigma_pop:float, p:float, xy_range:float, n:int, n_samples_base:int, min_n_samples:int, outlier_fraction:float,
                  test_run:bool=False, quick_run:bool=False):
    var_gnc_welsch = np.zeros((4,4))
    var_ls = np.zeros((4,4))

    n_samples = max(n_samples_base//n, min_n_samples)

    n0 = int((1.0-outlier_fraction)*n+0.5)
    ssize = int(math.ceil(math.sqrt(n0)))
    if not test_run:
        print("outlier_fraction=",outlier_fraction," n=",n," n0=",n0," n_samples=",n_samples,"sigma_pop=",sigma_pop,"xy_range=",xy_range,"ssize=",ssize)

    half_xy_range = 0.5*xy_range
    xy_scale = xy_range/(ssize-1)

    var_predicted = np.zeros((4,4))
    trs = TRS()
    for i in range(ssize):
        y = xy_scale*i - half_xy_range
        for j in range(ssize):
            idx = i*ssize+j
            if idx < n0:
                x = xy_scale*j - half_xy_range
                grad = trs.residual_gradient((x,y,0.0,0.0))
                gradT = np.matrix.transpose(grad)
                var_predicted += np.matmul(gradT,grad)

    var_predicted *= n/n0 # compensate for outlier ratio
    var_predicted = np.linalg.inv(var_predicted)
    var_predicted *= sigma_pop*sigma_pop

    for i in range(n_samples):
        model_gt = [randomM11(), randomM11(), randomM11(), randomM11()]
        data = np.zeros([n,4])
        for i in range(ssize):
            y = xy_scale*i - half_xy_range
            for j in range(ssize):
                idx = i*ssize+j
                if idx < n0:
                    x = xy_scale*j - half_xy_range
                    data[idx] = (x, y, model_gt[1]*x - model_gt[0]*y + model_gt[2] + np.random.normal(0.0, sigma_pop),
                                       model_gt[0]*x + model_gt[1]*y + model_gt[3] + np.random.normal(0.0, sigma_pop))

        # add outliers at random x positions in the same range
        for i in range(n0,n):
            data[i] = (half_xy_range*randomM11(), half_xy_range*randomM11(), 10.0*randomM11(), 10.0*randomM11())

        # GNC IRLS Welsch
        trs_instance = TRSWelsch(sigma_pop/p, max(xy_range,10.0*sigma_pop), 30, max_niterations=200)
        if trs_instance.run(data):
            trs_gnc_welsch = trs_instance.final_trs

        diff = trs_gnc_welsch-model_gt
        var_gnc_welsch += np.outer(diff, diff)

        # Least squares
        trs_ls = fit_trs_ls(data)
        diff = trs_ls-model_gt
        var_ls += np.outer(diff, diff)

    norm = 1.0/(n_samples-1)
    var_gnc_welsch = norm*var_gnc_welsch
    var_ls = norm*var_ls

    return var_predicted,var_gnc_welsch,var_ls

def main(test_run:bool, output_folder:str="../../../output", quick_run:bool=False):
    sigma_pop = 0.01
    p = 0.66666667

    # number of samples used for statistics
    n_samples_base = 100 if quick_run else 20000
    min_n_samples = 40 if quick_run else 1000

    np.random.seed(0) # We want the numbers to be the same on each run

    outlier_fraction_list = [0.0,0.1,0.2] if quick_run else [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for xy_range in [3.0]: #[3.0,5.0,10.0,30.0,100.0]:
        sample_size_array = [36] if quick_run else [100,225,400,900]
        for n in sample_size_array:
            eff_gnc_welsch_list = []
            eff_ls_list = []
            for outlier_fraction in outlier_fraction_list:
                var_predicted,var_gnc_welsch,var_ls = apply_to_data(sigma_pop, p, xy_range, n, n_samples_base, min_n_samples, outlier_fraction, test_run=test_run)
                fac = np.linalg.cholesky(var_gnc_welsch)
                facI = np.linalg.inv(fac)
                eff_mat = np.matmul(facI, np.matmul(var_predicted, np.matrix.transpose(facI)))
                eff = math.sqrt(np.linalg.det(eff_mat))
                if not test_run:
                    print("GNC Welsch estimator efficiency: ", eff)

                eff_gnc_welsch_list.append(eff)

                # Least squares
                fac = np.linalg.cholesky(var_ls)
                facI = np.linalg.inv(fac)
                eff_mat = np.matmul(facI, np.matmul(var_predicted, np.matrix.transpose(facI)))
                eff = math.sqrt(np.linalg.det(eff_mat))
                if not test_run:
                    print("Least squares efficiency: ", eff)

                eff_ls_list.append(eff)

            plt.close("all")
            plt.figure(num=1, dpi=240)
            plt.clf()
            ax = plt.gca()
            gncs_draw_curve(plt, eff_gnc_welsch_list, ("SupGN",  "Welsch", "GNC_Welsch"), xvalues=outlier_fraction_list)
            #gncs_draw_curve(plt, eff_ls_list,         ("LS",     "",       ""          ), xvalues=outlier_fraction_list)

            ax.set_xlabel(r'Outlier fraction' )
            ax.set_ylabel('Relative efficiency')
            #plt.box(False)
            ax.set_xlim(0.0,outlier_fraction_list[len(outlier_fraction_list)-1])
            ax.set_ylim(0.0,1.1)

            plt.legend()
            plt.savefig(os.path.join(output_folder, "trs_efficiency_n" + str(n) + "_range" + str(int(xy_range)) + ".png"), bbox_inches='tight')
            if not test_run:
                plt.show()

    if test_run:
        print("trs_efficiency OK")

if __name__ == "__main__":
    main(True, quick_run=False) # test_run
