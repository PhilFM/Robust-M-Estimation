import numpy as np
import matplotlib.pyplot as plt
import os
import math
from sklearn import linear_model

if __name__ == "__main__":
    import sys
    sys.path.append("../../../pypi_package/src")

from gnc_smoothie_philfm.plt_alg_vis import gncs_draw_curve
from gnc_smoothie_philfm.linear_model.linear_regressor_welsch import LinearRegressorWelsch

def fit_plane_ransac(data, sigma_pop: float):
    XYnp = np.array(data[:,0:2]).reshape((len(data),2))
    Znp = np.array(data[:,2])
    ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=1.5*sigma_pop, max_trials=2000)
    ransac.fit(X=XYnp, y=Znp)
    #inlier_mask = ransac.inlier_mask_
    coeff = ransac.estimator_.coef_
    #print("coeff=",coeff)
    intercept = ransac.estimator_.intercept_
    return np.array([coeff[0],coeff[1],intercept])

def fit_plane_ls(data):
    #print("data:",data)
    Sxx = Syy = Sxy = Sxz = Syz = Sx = Sy = Sz = 0.0
    for d in data:
        Sxx += d[0]*d[0]
        Sxy += d[0]*d[1]
        Syy += d[1]*d[1]
        Sxz += d[0]*d[2]
        Syz += d[1]*d[2]
        Sx += d[0]
        Sy += d[1]
        Sz += d[2]

    A = np.array([[Sxx,Sxy,Sx],[Sxy,Syy,Sy],[Sx,Sy,len(data)]])
    #print("A:",A)
    Ai = np.linalg.inv(A)
    #print("Ai:",Ai)
    return np.matmul(Ai, np.array([Sxz,Syz,Sz]))

def randomM11() -> float:
    return 2.0*(np.random.rand()-0.5)

def apply_to_data(sigma_pop:float, p:float, xy_range:float, n:int, n_samples_base:int, min_n_samples:int, outlier_fraction:float,
                  test_run:bool=False, quick_run:bool=False):
    var_gnc_welsch = np.zeros((3,3))
    var_ransac = np.zeros((3,3))
    var_ls = np.zeros((3,3))

    n_samples = max(n_samples_base//n, min_n_samples)

    n0 = int((1.0-outlier_fraction)*n+0.5)
    ssize = int(math.ceil(math.sqrt(n0)))
    if not test_run:
        print("outlier_fraction=",outlier_fraction," n=",n," n0=",n0," n_samples=",n_samples,"sigma_pop=",sigma_pop,"xy_range=",xy_range,"ssize=",ssize)

    half_xy_range = 0.5*xy_range
    xy_scale = xy_range/(ssize-1)

    var_predicted = np.zeros((3,3))
    for i in range(ssize):
        y = xy_scale*i - half_xy_range
        for j in range(ssize):
            idx = i*ssize+j
            if idx < n0:
                x = xy_scale*j - half_xy_range
                var_predicted[0][0] += x*x
                var_predicted[0][1] += x*y
                var_predicted[0][2] += x
                var_predicted[1][0] += x*y
                var_predicted[1][1] += y*y
                var_predicted[1][2] += y
                var_predicted[2][0] += x
                var_predicted[2][1] += y
                var_predicted[2][2] += 1.0

    var_predicted *= n/n0 # compensate for outlier ratio
    var_predicted = np.linalg.inv(var_predicted)
    var_predicted *= sigma_pop*sigma_pop #/(n0-1)

    for i in range(n_samples):
        plane_gt = [randomM11(), randomM11(), randomM11()]
        data = np.zeros((n,3))
        for i in range(ssize):
            y = xy_scale*i - half_xy_range
            for j in range(ssize):
                idx = i*ssize+j
                if idx < n0:
                    x = xy_scale*j - half_xy_range
                    data[idx] = (x, y, plane_gt[0]*x+plane_gt[1]*y+plane_gt[2] + np.random.normal(0.0, sigma_pop))

        # add outliers at random x positions in the same range
        for i in range(n0,n):
            data[i] = (half_xy_range*randomM11(), half_xy_range*randomM11(), 10.0*randomM11())

        # GNC IRLS Welsch
        plane_fitter = LinearRegressorWelsch(sigma_pop/p, max(xy_range,10.0*sigma_pop), 30, max_niterations=200)
        if plane_fitter.run(data):
            plane_gnc_welsch = plane_fitter.final_model

        diff = plane_gnc_welsch-plane_gt
        var_gnc_welsch += np.outer(diff, diff)

        # RANSAC
        plane_ransac = fit_plane_ransac(data, sigma_pop)
        diff = plane_ransac-plane_gt
        var_ransac += np.outer(diff, diff)

        # Least squares
        plane_ls = fit_plane_ls(data)
        diff = plane_ls-plane_gt
        var_ls += np.outer(diff, diff)

    norm = 1.0/(n_samples-1)
    var_gnc_welsch = norm*var_gnc_welsch
    var_ransac = norm*var_ransac
    var_ls = norm*var_ls

    return var_predicted,var_gnc_welsch,var_ransac,var_ls

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
            eff_ransac_list = []
            eff_ls_list = []
            for outlier_fraction in outlier_fraction_list:
                var_predicted,var_gnc_welsch,var_ransac,var_ls = apply_to_data(sigma_pop, p, xy_range, n, n_samples_base, min_n_samples, outlier_fraction, test_run=test_run)
                fac = np.linalg.cholesky(var_gnc_welsch)
                facI = np.linalg.inv(fac)
                eff_mat = np.matmul(facI, np.matmul(var_predicted, np.matrix.transpose(facI)))
                eff = math.sqrt(np.linalg.det(eff_mat))
                if not test_run:
                    print("GNC Welsch estimator efficiency: ", eff)

                eff_gnc_welsch_list.append(eff)

                # RANSAC
                fac = np.linalg.cholesky(var_ransac)
                facI = np.linalg.inv(fac)
                eff_mat = np.matmul(facI, np.matmul(var_predicted, np.matrix.transpose(facI)))
                eff = math.sqrt(np.linalg.det(eff_mat))
                if not test_run:
                    print("RANSAC efficiency: ", eff)

                eff_ransac_list.append(eff)

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
            gncs_draw_curve(plt, eff_ransac_list,     ("RANSAC", "",       ""          ), xvalues=outlier_fraction_list)
            #gncs_draw_curve(plt, eff_ls_list,         ("LS",     "",       ""          ), xvalues=outlier_fraction_list)

            ax.set_xlabel(r'Outlier fraction' )
            ax.set_ylabel('Relative efficiency')
            #plt.box(False)
            ax.set_xlim(0.0,outlier_fraction_list[len(outlier_fraction_list)-1])
            ax.set_ylim(0.0,1.1)

            plt.legend()
            plt.savefig(os.path.join(output_folder, "plane_fit_efficiency_n" + str(n) + "_range" + str(int(xy_range)) + ".png"), bbox_inches='tight')
            if not test_run:
                plt.show()

    if test_run:
        print("plane_fit_efficiency OK")

if __name__ == "__main__":
    main(False, quick_run=False) # test_run
