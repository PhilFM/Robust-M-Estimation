import numpy as np
from sklearn import linear_model
import cv2 as cv2
import sys
from robpy.utils.rho import TukeyBisquare
from robpy.regression import MMRegression
from sklearn.linear_model import LinearRegression
from statsmodels.api import add_constant

if __name__ == "__main__":
    sys.path.append("../../../pypi_package/src")

from gnc_smoothie.linear_model.linear_regressor_welsch import LinearRegressorWelsch
from gnc_smoothie.linear_model.linear_regressor_pseudo_huber import LinearRegressorPseudoHuber
from gnc_smoothie.linear_model.linear_regressor_gnc_irls_p import LinearRegressorGNC_IRLSp

def fit_regression_ls(data_x, data_y, weight):
    #print("      LS start")
    Xp = np.sqrt(weight).reshape(-1, 1)*add_constant(data_x, prepend=False)
    yp = (np.sqrt(weight)*data_y).reshape(-1,1)
    #print("X=",Xp)
    #print("y=",yp)
    model = LinearRegression(fit_intercept=False).fit(Xp, yp)
    #print("      LS end")
    return model.coef_

def fit_regression_gnc_welsch(data_x:np.ndarray, data_y:np.ndarray, sigma_base:float):
    #print("      GNCW start")
    regressor = LinearRegressorWelsch(sigma_base=sigma_base, sigma_limit=max(data_y)-min(data_y), max_niterations=200)#, debug=True, messages_file=sys.stdout)
    regressor.fit((data_x, data_y)) # may fail but use the latest value whatever happened
    #print("final_coeff=",regressor.final_coeff,"final_intercept=",regressor.final_intercept)
    #print("      GNCW end")
    return np.concatenate((regressor.final_coeff[0],regressor.final_intercept))

def fit_regression_mm_estimation(data_x:np.ndarray, data_y:np.ndarray, c:float):
    #print("      MM start")
    estimator = MMRegression(prepend_intercept=False, rho=TukeyBisquare(c)).fit(data_x, data_y)
    #print("      MM end")
    return estimator.model.coef_

def fit_regression_huber(data_x:np.ndarray, data_y:np.ndarray, sigma:float):
    #print("      Huber start")
    line_fitter = LinearRegressorPseudoHuber(sigma, max_niterations=200)
    line_fitter.fit((data_x, data_y)) # may fail but use the latest value whatever happened
    #print("      Huber end")
    return line_fitter.final_coeff,line_fitter.final_intercept

def fit_regression_gnc_irls_p(data_x:np.ndarray, data_y:np.ndarray, sigma:float, x_range:float):
    #print("      GNC IRLS-p start")
    gnc_irls_p_p = 0.0
    gnc_irls_p_rscale = 1.0/x_range
    gnc_irls_p_epsilon_base = gnc_irls_p_rscale*sigma
    gnc_irls_p_epsilon_limit = 1.0
    gnc_irls_p_beta = 0.8

    line_fitter = LinearRegressorGNC_IRLSp(gnc_irls_p_p, gnc_irls_p_rscale, gnc_irls_p_epsilon_base, gnc_irls_p_epsilon_limit, gnc_irls_p_beta, max_niterations=200)
    line_fitter.fit((data_x, data_y)) # may fail but use the latest value whatever happened
    #print("      GNC IRLS-p end")
    return line_fitter.final_coeff,line_fitter.final_intercept
    
def fit_regression_theil_sen(data_x:np.ndarray, data_y:np.ndarray):
    #print("      Theil-Sen start")
    #Xnp = np.array(data[:,0]).reshape((len(data),1))
    #Ynp = np.array(data[:,1])
    theil_sen = linear_model.TheilSenRegressor() #max_subpopulation=1e10)
    theil_sen.fit(X=data_x, y=data_y)
    #inlier_mask = theil_sen.inlier_mask_
    coeff = theil_sen.coef_
    intercept = theil_sen.intercept_
    #print("      Theil-Sen end")
    return np.concatenate((coeff,np.array([intercept])))

def fit_regression_ransac(data_x:np.ndarray, data_y:np.ndarray, sigma_pop: float):
    #print("      RANSAC start")
    ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=1.5*sigma_pop, max_trials=2000)
    ransac.fit(X=data_x, y=data_y)
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

        return fit_regression_ls(ls_data)

    #print("coeff=",coeff)
    #print("intercept=",intercept)
    #print("      RANSAC end")
    return np.concatenate((coeff,[intercept]))
