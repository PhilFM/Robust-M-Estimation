import numpy as np
from sklearn import linear_model
import cv2 as cv2
import sys
from robpy.utils.rho import TukeyBisquare
from robpy.regression import MMRegression

if __name__ == "__main__":
    sys.path.append("../../../pypi_package/src")

from gnc_smoothie.linear_model.linear_regressor_welsch import LinearRegressorWelsch
from gnc_smoothie.linear_model.linear_regressor_pseudo_huber import LinearRegressorPseudoHuber
from gnc_smoothie.linear_model.linear_regressor_gnc_irls_p import LinearRegressorGNC_IRLSp

def fit_line_ls(data):
    Sxx = Sx = Sy = Sxy = 0.0
    for d in data:
        Sxx += d[0]*d[0]
        Sx += d[0]
        Sy += d[1]
        Sxy += d[0]*d[1]

    return np.matmul(np.linalg.inv(np.array([[Sxx,Sx],[Sx,len(data)]])), np.array([Sxy,Sy]))

def fit_line_gnc_welsch(data:np.ndarray, sigma_base:float, sigma_limit:float):
    line_fitter = LinearRegressorWelsch(sigma_base=sigma_base, sigma_limit=sigma_limit, num_sigma_steps=30, max_niterations=200)
    line_fitter.fit(data) # may fail but use the latest value whatever happened
    return line_fitter.final_model

def fit_line_huber(data:np.ndarray, sigma:float):
    line_fitter = LinearRegressorPseudoHuber(sigma, max_niterations=200)
    line_fitter.fit(data) # may fail but use the latest value whatever happened
    return line_fitter.final_model

def fit_line_gnc_irls_p(data:np.ndarray, sigma:float, x_range:float):
    gnc_irls_p_p = 0.0
    gnc_irls_p_rscale = 1.0/x_range
    gnc_irls_p_epsilon_base = gnc_irls_p_rscale*sigma
    gnc_irls_p_epsilon_limit = 1.0
    gnc_irls_p_beta = 0.8

    line_fitter = LinearRegressorGNC_IRLSp(gnc_irls_p_p, gnc_irls_p_rscale, gnc_irls_p_epsilon_base, gnc_irls_p_epsilon_limit, gnc_irls_p_beta, max_niterations=200)
    line_fitter.fit(data) # may fail but use the latest value whatever happened
    return line_fitter.final_model
    
def fit_line_theil_sen(data):
    Xnp = np.array(data[:,0]).reshape((len(data),1))
    Ynp = np.array(data[:,1])
    theil_sen = linear_model.TheilSenRegressor() #max_subpopulation=1e10)
    theil_sen.fit(X=Xnp, y=Ynp)
    #inlier_mask = theil_sen.inlier_mask_
    coeff = theil_sen.coef_
    intercept = theil_sen.intercept_
    return np.array([coeff[0],intercept])

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

def fit_line_hough(data, sigma_pop: float, max_rho: float, test_run: bool) -> np.ndarray:
    #print("data=",data)
    datap = data.reshape(-1, 1, 2).astype(np.float32)
    #print("datap=",datap)
    lines = cv2.HoughLinesPointSet(datap, lines_max=1, threshold=0, min_rho=-max_rho, max_rho=max_rho,
                                   rho_step=0.01*max_rho, min_theta=0.0, max_theta=np.pi, 
                                   theta_step=0.005*np.pi)
    #print("lines=",lines)

    _, rho, theta = lines[:, 0][:, 0], lines[:, 0][:, 1], lines[:, 0][:, 2]

    # Convert to cartesian
    theta[theta == 0.] = 1e-5  # to avoid division by 0 in next line
    a = -1 / np.tan(theta)  # the implied lines are perpendicular to theta
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    b = y - a * x
    return np.array([a[0],b[0]])
