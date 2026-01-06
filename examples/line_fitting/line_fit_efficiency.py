import numpy as np
import matplotlib.pyplot as plt
import os
import math
from sklearn import linear_model
import cv2 as cv2

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

from gnc_smoothie_philfm.plt_alg_vis import gncs_draw_curve
from gnc_smoothie_philfm.linear_model.linear_regressor_welsch import LinearRegressorWelsch

def fit_line_ls(data):
    Sxx = Sx = Sy = Sxy = 0.0
    for d in data:
        Sxx += d[0]*d[0]
        Sx += d[0]
        Sy += d[1]
        Sxy += d[0]*d[1]

    return np.matmul(np.linalg.inv(np.array([[Sxx,Sx],[Sx,len(data)]])), np.array([Sxy,Sy]))

def fit_line_ransac(data, sigma_pop: float):
    Xnp = np.array(data[:,0]).reshape((len(data),1))
    Ynp = np.array(data[:,1])
    ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=1.5*sigma_pop, max_trials=2000)
    ransac.fit(X=Xnp, y=Ynp)
    inlier_mask = ransac.inlier_mask_
    coeff = ransac.estimator_.coef_
    intercept = ransac.estimator_.intercept_

    # it seems that RANSACRegressor applies least squares to the inliers,
    # so we don't need the following code
    if False:
        # optimise with least squares
        ls_data = []
        for i,d in enumerate(data):
            if inlier_mask[i]:
                ls_data.append(d)

        return fit_line_ls(ls_data)

    return np.array([coeff[0],intercept])

def fit_line_hough(data, sigma_pop: float, test_run: bool) -> np.ndarray:
    #print("data=",data)
    datap = data.reshape(-1, 1, 2).astype(np.float32)
    #print("datap=",datap)
    lines = cv2.HoughLinesPointSet(datap, lines_max=1, threshold=0, min_rho=-1.0, max_rho=1.0, 
                                   rho_step=0.01, min_theta=0.0, max_theta=np.pi, 
                                   theta_step=np.pi/200)
    #print("lines=",lines)

    votes, rho, theta = lines[:, 0][:, 0], lines[:, 0][:, 1], lines[:, 0][:, 2]
    #if not test_run:
    #    print("votes=",votes)

    # Convert to cartesian
    theta[theta == 0.] = 1e-5  # to avoid division by 0 in next line
    a = -1 / np.tan(theta)  # the implied lines are perpendicular to theta
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    b = y - a * x
    return np.array([a[0],b[0]])

def randomM11() -> float:
    return 2.0*(np.random.rand()-0.5)

def apply_to_data(sigma_pop,p,x_range,n,n_samples_base,min_n_samples,outlier_fraction,
                  test_run:bool=False, quick_run:bool=False):
    var_gnc_welsch = np.zeros((2,2))
    var_ransac = np.zeros((2,2))
    var_hough = np.zeros((2,2))
    var_ls = np.zeros((2,2))

    n0 = int((1.0-outlier_fraction)*n+0.5)
    n_samples = max(n_samples_base//n, min_n_samples)
    if not test_run:
        print("outlier_fraction=",outlier_fraction," n=",n," n0=",n0," n_samples=",n_samples,"sigma_pop=",sigma_pop,"x_range=",x_range)

    half_x_range = 0.5*x_range
    x_scale = x_range/(n0-1)

    var_predicted = np.zeros((2,2))
    for i in range(n0):
        x = x_scale*i - half_x_range
        var_predicted[0][0] += x*x
        var_predicted[0][1] += x
        var_predicted[1][0] += x
        var_predicted[1][1] += 1.0

    var_predicted *= n/n0 # compensate for outlier ratio
    var_predicted = np.linalg.inv(var_predicted)
    var_predicted *= sigma_pop*sigma_pop #/(n0-1)

    for i in range(n_samples):
        line_gt = [randomM11(), randomM11()]
        #print("line_gt=",line_gt)
        data = np.zeros((n,2))
        for i in range(n0):
            x = x_scale*i - half_x_range
            data[i] = (x, line_gt[0]*x+line_gt[1] + np.random.normal(0.0, sigma_pop))

        # add outliers at random x positions in the same range
        for i in range(n0,n):
            data[i] = (half_x_range*randomM11(), 10.0*randomM11())

        # GNC IRLS Welsch
        line_fitter = LinearRegressorWelsch(sigma_pop/p, max(x_range,10.0*sigma_pop), 30, max_niterations=200)
        if line_fitter.run(data):
            coeff = line_fitter.final_coeff
            intercept = line_fitter.final_intercept
            line_gnc_welsch = np.array([coeff[0][0], intercept[0]])

        diff = line_gnc_welsch-line_gt
        var_gnc_welsch += np.outer(diff, diff)

        # RANSAC
        line_ransac = fit_line_ransac(data, sigma_pop)
        diff = line_ransac-line_gt
        var_ransac += np.outer(diff, diff)

        # Hough transform
        line_hough = fit_line_hough(data, sigma_pop, test_run)
        diff = line_hough-line_gt
        #print("diff=",diff)
        var_hough += np.outer(diff, diff)

        # Least squares
        line_ls = fit_line_ls(data)
        diff = line_ls-line_gt
        var_ls += np.outer(diff, diff)

    norm = 1.0/(n_samples-1)
    var_gnc_welsch = norm*var_gnc_welsch
    var_ransac = norm*var_ransac
    var_hough = norm*var_hough
    var_ls = norm*var_ls
    return var_predicted,var_gnc_welsch,var_ransac,var_hough,var_ls

def main(test_run:bool, output_folder:str="../../output", quick_run:bool=False):
    sigma_pop = 0.01
    p = 0.66666667

    # number of samples used for statistics
    n_samples_base = 100 if quick_run else 20000
    min_n_samples = 40 if quick_run else 1000

    np.random.seed(0) # We want the numbers to be the same on each run

    outlier_fraction_list = [0.0,0.3,0.6] if quick_run else [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for x_range in [3.0]: #[3.0,5.0,10.0,30.0,100.0]:
        sample_size_array = [50] if quick_run else [30,100,300,1000]
        for n in sample_size_array:
            eff_gnc_welsch_list = []
            eff_ransac_list = []
            eff_hough_list = []
            eff_ls_list = []
            for outlier_fraction in outlier_fraction_list:
                var_predicted,var_gnc_welsch,var_ransac,var_hough,var_ls = apply_to_data(sigma_pop, p, x_range, n, n_samples_base, min_n_samples, outlier_fraction, test_run=test_run)

                # GNC IRLS Welsch
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

                # Hough transform
                fac = np.linalg.cholesky(var_hough)
                facI = np.linalg.inv(fac)
                eff_mat = np.matmul(facI, np.matmul(var_predicted, np.matrix.transpose(facI)))
                eff = math.sqrt(np.linalg.det(eff_mat))
                if not test_run:
                    print("Hough transform efficiency: ", eff)

                eff_hough_list.append(eff)

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
            gncs_draw_curve(plt, eff_hough_list,      ("Hough",  "",       ""          ), xvalues=outlier_fraction_list)
            #gncs_draw_curve(plt, eff_ls_list,         ("LS",     "",       ""          ), xvalues=outlier_fraction_list)

            ax.set_xlabel(r'Outlier fraction' )
            ax.set_ylabel('Relative efficiency')
            #plt.box(False)
            ax.set_xlim(0.0,outlier_fraction_list[len(outlier_fraction_list)-1])
            ax.set_ylim(0.0,1.1)

            plt.legend()
            plt.savefig(os.path.join(output_folder, "line_fit_efficiency_n" + str(n) + "_range" + str(int(x_range)) + ".png"), bbox_inches='tight')
            if not test_run:
                plt.show()

    if test_run:
        print("line_fit_efficiency OK")

if __name__ == "__main__":
    main(False, quick_run=False) # test_run
