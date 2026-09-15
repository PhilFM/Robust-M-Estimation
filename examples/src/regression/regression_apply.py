import numpy as np
import math
import time
from statsmodels.api import add_constant

from fit_regression import fit_regression_ls,fit_regression_gnc_welsch,fit_regression_huber,fit_regression_gnc_irls_p,fit_regression_mm_estimation,fit_regression_theil_sen,fit_regression_ransac

def randomM11() -> float:
    return 2.0*(np.random.rand()-0.5)

class CompareRegressionResult:
    def __init__(
            self,
            ndim:int = 10
    ):
        # variances
        self.var_ls = np.zeros((ndim,ndim))
        self.var_gnc_welsch = np.zeros((ndim,ndim))
        self.var_mm_estimation = np.zeros((ndim,ndim))
        self.var_huber = np.zeros((ndim,ndim))
        self.var_gnc_irls_p = np.zeros((ndim,ndim))
        self.var_theil_sen = np.zeros((ndim,ndim))
        self.var_ransac = np.zeros((ndim,ndim))

        # timing
        self.av_time_ls = 0.0
        self.av_time_gnc_welsch = 0.0
        self.av_time_mm_estimation = 0.0
        self.av_time_huber = 0.0
        self.av_time_gnc_irls_p = 0.0
        self.av_time_theil_sen = 0.0
        self.av_time_ransac = 0.0
    
        self.n_samples = None

def apply_to_data(ndim:int,
                  sigma_pop:float,
                  welsch_q:float,
                  huber_sigma_scale:float,
                  gnc_irls_p_epsilon_scale:float,
                  x_range:float,
                  n_data_points:int,
                  n_samples_base:int,
                  min_n_samples:int,
                  outlier_fraction:float,
                  test_run:bool=False,
                  quick_run:bool=False) -> CompareRegressionResult:
    alg_result = CompareRegressionResult(ndim)
    alg_result.n_samples = max(n_samples_base//n_data_points, min_n_samples)

    if not test_run:
        print("ndim=",ndim,"outlier_fraction=",outlier_fraction,"n_data_points=",n_data_points,"n_samples=",alg_result.n_samples,"sigma_pop=",sigma_pop,"x_range=",x_range)

    half_x_range = 0.5*x_range
    n_outliers = int(0.5+outlier_fraction*n_data_points)

    for i in range(alg_result.n_samples):
        np.random.seed(i*213)

        model_gt = np.zeros(ndim)
        for i in range(ndim):
            model_gt[i] = randomM11()

        data_x = np.zeros((n_data_points,ndim-1))
        data_y = np.zeros(n_data_points)
        for i in range(n_data_points):
            tot = 0.0
            for j in range(ndim-1):
                data_x[i][j] = randomM11()*half_x_range
                tot += model_gt[j]*data_x[i][j]

            data_y[i] = tot + model_gt[ndim-1] + np.random.normal(0.0, sigma_pop)

        # add outliers at random x positions in the same range
        outlier_list = np.random.randint(n_data_points, size=n_outliers)
        for i in outlier_list:
            #print("outlier_idx=",i)
            data_y[i] += 10.0*randomM11()

        # GNC IRLS Welsch
        start_time = time.process_time()
        model_est = fit_regression_gnc_welsch(data_x, data_y, sigma_pop/welsch_q)
        alg_result.av_time_gnc_welsch += time.process_time()-start_time
        diff = model_est-model_gt
        alg_result.var_gnc_welsch += np.outer(diff, diff)

        # Pseudo-Huber
        start_time = time.process_time()
        coeff_est,intercept_est = fit_regression_huber(data_x, data_y, sigma_pop*huber_sigma_scale)
        model_est = np.append(coeff_est[0], intercept_est)
        alg_result.av_time_huber += time.process_time()-start_time
        diff = model_est-model_gt
        alg_result.var_huber += np.outer(diff, diff)

        # GNC IRLS-p
        start_time = time.process_time()
        coeff_est,intercept_est = fit_regression_gnc_irls_p(data_x, data_y, sigma_pop*gnc_irls_p_epsilon_scale, x_range)
        model_est = np.append(coeff_est[0], intercept_est)
        alg_result.av_time_gnc_irls_p += time.process_time()-start_time
        diff = model_est-model_gt
        alg_result.var_gnc_irls_p += np.outer(diff, diff)

        if False:
            dx = add_constant(data_x, prepend=False)
            dxc = np.matmul(dx,model_est)
            print("dxc.shape=",dxc.shape)
            residuals = dxc - data_y
            av_err = np.sum(residuals ** 2)/(sigma_pop*sigma_pop*n_data_points)
            print("av_err=",av_err)

        # MM estimation
        start_time = time.process_time()
        model_est = fit_regression_mm_estimation(data_x, data_y, 3.44)
        alg_result.av_time_mm_estimation += time.process_time()-start_time
        diff = model_est-model_gt
        alg_result.var_mm_estimation += np.outer(diff, diff)

        # Theil-Sen
        start_time = time.process_time()
        model_est = fit_regression_theil_sen(data_x, data_y)
        alg_result.av_time_theil_sen += time.process_time()-start_time
        diff = model_est-model_gt
        #print("      Theil-Sen result=",model_est, "error=",diff)
        alg_result.var_theil_sen += np.outer(diff, diff)

        # RANSAC
        start_time = time.process_time()
        model_est = fit_regression_ransac(data_x, data_y, sigma_pop)
        alg_result.av_time_ransac += time.process_time()-start_time
        diff = model_est-model_gt
        #print("      RANSAC result=",model_est, "error=",diff)
        alg_result.var_ransac += np.outer(diff, diff)

        # Least squares with outliers removed
        ls_weight = np.ones(n_data_points)
        for i in outlier_list:
            ls_weight[i] = 0.0

        start_time = time.process_time()
        model_est = fit_regression_ls(data_x, data_y, ls_weight)
        alg_result.av_time_ls += time.process_time()-start_time
        diff = model_est-model_gt
        #print("      LS diff=",diff)
        alg_result.var_ls += np.outer(diff, diff)

    norm = 1.0/(alg_result.n_samples-1)

    alg_result.var_gnc_welsch *= norm
    alg_result.var_huber *= norm
    alg_result.var_gnc_irls_p *= norm
    alg_result.var_mm_estimation *= norm
    alg_result.var_theil_sen *= norm
    alg_result.var_ransac *= norm
    alg_result.var_ls *= norm
    alg_result.av_time_gnc_welsch /= alg_result.n_samples
    alg_result.av_time_huber /= alg_result.n_samples
    alg_result.av_time_gnc_irls_p /= alg_result.n_samples
    alg_result.av_time_mm_estimation /= alg_result.n_samples
    alg_result.av_time_theil_sen /= alg_result.n_samples
    alg_result.av_time_ransac /= alg_result.n_samples
    alg_result.av_time_ls /= alg_result.n_samples
    return alg_result
