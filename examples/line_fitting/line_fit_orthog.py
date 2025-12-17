import numpy as np
import math

# Line model is a*x + b*y + c = 0, a^2 + b^2 = 1
# This class is for orthogonal regression
class LineFitOrthog:
    def __init__(self):
        pass

    # copy model parameters and apply any internal calculations
    def cache_model(self, model, model_ref=None):
        norm = math.sqrt(model[0]*model[0] + model[1]*model[1])
        self.__a = model[0]/norm
        self.__b = model[1]/norm
        self.__c = model[2]/norm

    # r = a*xi + b*y + c*z + d
    def residual(self, data_item) -> np.array:
        x = data_item[0]
        y = data_item[1]
        return np.array([self.__a*x + self.__b*y + self.__c])

    def __weighted_mean(self, data, weight, scale):
        X = np.zeros(2)
        sW = 0.0
        for d,w,s in zip(data, weight, scale, strict=True):
            w /= s * s
            X += w*d
            sW += w

        return X / sW

    # fits the model to the data
    def weighted_fit(self, data, weight, scale) -> (np.array, np.array):

        # calculate weighted mean
        X0 = self.__weighted_mean(data, weight, scale)

        # calculate covariance matrix, subtracting the weighted mean from each point
        cov = np.zeros((2,2))
        for d,w,s in zip(data, weight, scale, strict=True):
            dp = d-X0
            w /= s * s
            cov += w*np.outer(dp,dp)

        # calculate normal vector as smallest eigenvalue of the covariance matrix
        e_val, e_vect = np.linalg.eig(cov)
        min_eval = np.argmin(e_val)
        normal_vector = e_vect[:, min_eval]
        return np.array([normal_vector[0], normal_vector[1], -np.dot(normal_vector,X0)]),None
