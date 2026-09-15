import numpy as np
import math
import time

from fit_line import fit_line_ls,fit_line_gnc_welsch,fit_line_huber,fit_line_gnc_irls_p,fit_line_theil_sen,fit_line_ransac,fit_line_hough

def randomM11() -> float:
    return 2.0*(np.random.rand()-0.5)

class CompareLineAlgResult:
    def __init__(
        self
    ):
        # variances
        self.var_predicted = np.zeros((2,2))
        self.var_ls = np.zeros((2,2))
        self.var_gnc_welsch = np.zeros((2,2))
        self.var_huber = np.zeros((2,2))
        self.var_gnc_irls_p = np.zeros((2,2))
        self.var_theil_sen = np.zeros((2,2))
        self.var_ransac = np.zeros((2,2))
        self.var_hough = np.zeros((2,2))

        # timing
        self.av_time_ls = 0.0
        self.av_time_gnc_welsch = 0.0
        self.av_time_huber = 0.0
        self.av_time_gnc_irls_p = 0.0
        self.av_time_theil_sen = 0.0
        self.av_time_ransac = 0.0
        self.av_time_hough = 0.0
    
        self.n_samples = None

def apply_to_data(show_known_scale_results: bool, sigma_pop:float, welsch_q:float, huber_sigma_scale:float, gnc_irls_p_epsilon_scale:float, x_range,n,n_samples_base,min_n_samples,outlier_fraction,
                  test_run:bool=False, quick_run:bool=False) -> CompareLineAlgResult:
    alg_result = CompareLineAlgResult()
    alg_result.n_samples = max(n_samples_base//n, min_n_samples)

    n0 = int((1.0-outlier_fraction)*n+0.5)
    if not test_run:
        print("outlier_fraction=",outlier_fraction," n=",n," n0=",n0," n_samples=",alg_result.n_samples,"sigma_pop=",sigma_pop,"x_range=",x_range)

    half_x_range = 0.5*x_range
    x_scale = x_range/(n0-1)

    for i in range(n0):
        x = x_scale*i - half_x_range
        alg_result.var_predicted[0][0] += x*x
        alg_result.var_predicted[0][1] += x
        alg_result.var_predicted[1][0] += x
        alg_result.var_predicted[1][1] += 1.0

    alg_result.var_predicted *= n/n0 # compensate for outlier ratio
    alg_result.var_predicted = np.linalg.inv(alg_result.var_predicted)
    alg_result.var_predicted *= sigma_pop*sigma_pop #/(n0-1)

    for i in range(alg_result.n_samples):
        np.random.seed(i*213)

        line_gt = [randomM11(), randomM11()]
        #print("line_gt=",line_gt)
        data = np.zeros((n,2))
        for i in range(n0):
            x = x_scale*i - half_x_range
            data[i] = (x, line_gt[0]*x+line_gt[1] + np.random.normal(0.0, sigma_pop))

        # add outliers at random x positions in the same range
        for i in range(n0,n):
            data[i] = (half_x_range*randomM11(), 10.0*randomM11())

        #print("data=",data)
        if show_known_scale_results:
            # GNC IRLS Welsch
            start_time = time.time()
            line_est = fit_line_gnc_welsch(data, sigma_pop/welsch_q, max(x_range,10.0*sigma_pop))
            alg_result.av_time_gnc_welsch += time.time()-start_time
            diff = line_est-line_gt
            alg_result.var_gnc_welsch += np.outer(diff, diff)

            # Pseudo-Huber
            start_time = time.time()
            line_est = fit_line_huber(data, sigma_pop*huber_sigma_scale)
            alg_result.av_time_huber += time.time()-start_time
            diff = line_est-line_gt
            alg_result.var_huber += np.outer(diff, diff)

            # GNC IRLS-p
            start_time = time.time()
            line_est = fit_line_gnc_irls_p(data, sigma_pop*gnc_irls_p_epsilon_scale, x_range)
            alg_result.av_time_gnc_irls_p += time.time()-start_time
            diff = line_est-line_gt
            alg_result.var_gnc_irls_p += np.outer(diff, diff)

        # Theil-Sen
        start_time = time.time()
        line_est = fit_line_theil_sen(data)
        alg_result.av_time_theil_sen += time.time()-start_time
        diff = line_est-line_gt
        alg_result.var_theil_sen += np.outer(diff, diff)

        # RANSAC
        start_time = time.time()
        line_est = fit_line_ransac(data, sigma_pop)
        alg_result.av_time_ransac += time.time()-start_time
        diff = line_est-line_gt
        alg_result.var_ransac += np.outer(diff, diff)

        # Hough transform
        start_time = time.time()
        line_est = fit_line_hough(data, sigma_pop, half_x_range, test_run)
        alg_result.av_time_hough += time.time()-start_time
        diff = line_est-line_gt
        alg_result.var_hough += np.outer(diff, diff)

        # Least squares
        start_time = time.time()
        line_est = fit_line_ls(data)
        alg_result.av_time_ls += time.time()-start_time
        diff = line_est-line_gt
        alg_result.var_ls += np.outer(diff, diff)

    norm = 1.0/(alg_result.n_samples-1)
    if show_known_scale_results:
        alg_result.var_gnc_welsch *= norm
        alg_result.var_huber *= norm
        alg_result.var_gnc_irls_p *= norm

    alg_result.var_theil_sen *= norm
    alg_result.var_ransac *= norm
    alg_result.var_hough *= norm
    alg_result.var_ls *= norm
    if show_known_scale_results:
        alg_result.av_time_gnc_welsch /= alg_result.n_samples
        alg_result.av_time_huber /= alg_result.n_samples
        alg_result.av_time_gnc_irls_p /= alg_result.n_samples

    alg_result.av_time_theil_sen /= alg_result.n_samples
    alg_result.av_time_ransac /= alg_result.n_samples
    alg_result.av_time_hough /= alg_result.n_samples
    alg_result.av_time_ls /= alg_result.n_samples
    return alg_result
