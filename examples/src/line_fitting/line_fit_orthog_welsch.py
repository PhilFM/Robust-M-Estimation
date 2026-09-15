import numpy as np
import math
import sys

from gnc_smoothie.irls import IRLS
from gnc_smoothie.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie.linear_model.linear_regressor_welsch import LinearRegressorWelsch

sys.path.append("../misc")
from circular_median import circular_median_radians

from line_fit_orthog import LineFitOrthog

class LineFitOrthogWelsch:
    def __init__(
            self,
            sigma: float,
            *,
            sigma_limit: float = 20.0,
            num_sigma_steps: int = 20,
            max_niterations: int = 50,
            diff_thres: float = 1.e-10,
            messages_file = None,
            debug: bool = False
            ):
        self.__sigma = sigma
        self.__sigma_limit = sigma_limit
        self.__num_sigma_steps = num_sigma_steps
        self.__max_niterations = max_niterations
        self.__diff_thres = diff_thres
        self.__messages_file = messages_file
        self.__debug = debug

    def __convert_model(self, model: np.array) -> np.array:
        return model/math.sqrt(model[0]*model[0]+model[1]*model[1])

    def __convert_to_orthog(self, coeff, intercept, angle:float, y_offset:float):
        # xd = x2-x1, yd = y2-y1
        # angle = atan2(yd,xd)
        # So xd = d*ca, yd = d*sa
        #   (x') = ( ca sa) (x) + (  0  )
        #   (y')   (-sa ca) (y)   (-yoff)
        # So
        #   (x) = (ca -sa) (x'     )
        #   (y)   (sa  ca) (y'+yoff)
        # We have y' = a'*x' + b'
        # So this gives
        #   -sa*x + ca*y - yoff = a'*(ca*x + sa*y) + b'
        # To get a*x + b*y + c we need
        #   a = a'*ca+sa, b = a'*sa-ca, c = b'+yoff
        cosa = math.cos(angle)
        sina = math.sin(angle)
        #print("coeff=",coeff,"intercept=",intercept)
        return self.__convert_model(np.array([coeff[0][0]*cosa+sina, coeff[0][0]*sina-cosa, intercept[0]+y_offset]))

    def fit(self,
            data,
            *,
            method: str = "BestAngle", # also could be "Median", "IRLS"
            n_angles: int = 5,
            weight: np.array = None,
            scale: np.array = None):

        if method == "BestAngle":
            # build list of angles
            angle_list = np.linspace(0.0, math.pi*(1.0-1.0/n_angles), n_angles)
            data_x = np.zeros(len(data))
            data_y = np.zeros(len(data))
            max_objective_val = 0.0
            for angle in angle_list:
                cosa = math.cos(angle)
                sina = math.sin(angle)

                # we will calculate the correct y offset later - only the angle is relevant at this stage
                y_offset = 0.0 

                # build new data array rotated appropriately
                for i, d in enumerate(data):
                    data_x[i] =  d[0]*cosa + d[1]*sina
                    data_y[i] = -d[0]*sina + d[1]*cosa - y_offset

                # linear regression fitter y = a*x + b
                y_range = max(data_y) - min(data_y)
                line_fitter = LinearRegressorWelsch(sigma_base=self.__sigma, sigma_limit=min(y_range, self.__sigma_limit),
                                                    num_sigma_steps=self.__num_sigma_steps,
                                                    max_niterations=self.__max_niterations,
                                                    diff_thres=self.__diff_thres,
                                                    messages_file=self.__messages_file,
                                                    debug=self.__debug)
                if line_fitter.fit((data_x, data_y), weight=weight, scale=scale):
                    if line_fitter.final_objective_val > max_objective_val:
                        self.final_line = self.__convert_to_orthog(line_fitter.final_coeff, line_fitter.final_intercept, angle, y_offset)
                        self.final_objective_val = max_objective_val = line_fitter.final_objective_val
                        self.final_weight = line_fitter.final_weight
                        if self.__debug:
                            # calculate y offset as weighted mean
                            weight_tot = 0.0
                            y_tot = 0.0
                            for d,w in zip(data,line_fitter.final_weight, strict=True):
                                weight_tot += w
                                y_tot += w*(-d[0]*sina + d[1]*cosa)

                            self.debug_ref_line = self.__convert_to_orthog([[0]], [0], angle, y_tot/weight_tot)
                            self.debug_line_list = [(model[0], self.__convert_to_orthog(model[1][0], model[1][1], angle, y_offset), model[2], model[3]) for model in line_fitter.debug_model_list]
                            self.debug_weighted_derivs_time = line_fitter.debug_weighted_derivs_time
                            self.debug_solve_time = line_fitter.debug_solve_time
                            self.debug_total_time = line_fitter.debug_total_time
                            self.debug_n_iterations = line_fitter.debug_n_iterations

            return True if max_objective_val > 0.0 else False
        elif method == "Median":
            # build list of angles
            angle_list = []
            for i, d in enumerate(data):
                for j in range(i+1,len(data)):
                    angle_list.append(math.atan2(d[1]-data[j][1], d[0]-data[j][0]))

            angle = circular_median_radians(angle_list)
            cosa = math.cos(angle)
            sina = math.sin(angle)
            #print("angle=",angle,"cosa=",cosa,"sina=",sina)

            # calculate intercept using median
            y_offset_list = []
            for d in data:
                y_offset_list.append(-d[0]*sina + d[1]*cosa)

            y_offset = np.median(y_offset_list)

            #self.final_line = self.__convert_to_orthog([[0]], [0], angle, y_offset)
            #self.final_weight = np.ones(len(data))
            #self.debug_line_list = ()
            #return True

            # build new data array rotated appropriately
            data_x = np.zeros(len(data))
            data_y = np.zeros(len(data))
            for i, d in enumerate(data):
                data_x[i] =  d[0]*cosa + d[1]*sina
                data_y[i] = -d[0]*sina + d[1]*cosa - y_offset

            #print("data_x=",data_x)
            #print("data_y=",data_y)

            # linear regression fitter y = a*x + b
            y_range = max(data_y) - min(data_y)
            line_fitter = LinearRegressorWelsch(sigma_base=self.__sigma, sigma_limit=min(y_range, self.__sigma_limit),
                                                num_sigma_steps=self.__num_sigma_steps,
                                                max_niterations=self.__max_niterations,
                                                diff_thres=self.__diff_thres,
                                                messages_file=self.__messages_file,
                                                debug=self.__debug)
            if line_fitter.fit((data_x, data_y), weight=weight, scale=scale):
                self.final_line = self.__convert_to_orthog(line_fitter.final_coeff, line_fitter.final_intercept, angle, y_offset)
                self.final_weight = line_fitter.final_weight
                if self.__debug:
                    self.debug_theil_sen_line = self.__convert_to_orthog([[0]], [0], angle, y_offset)
                    self.debug_line_list = [(model[0], self.__convert_to_orthog(model[1][0], model[1][1], angle, y_offset), model[2], model[3]) for model in line_fitter.debug_model_list]
                    self.debug_weighted_derivs_time = line_fitter.debug_weighted_derivs_time
                    self.debug_solve_time = line_fitter.debug_solve_time
                    self.debug_total_time = line_fitter.debug_total_time
                    self.debug_n_iterations = line_fitter.debug_n_iterations

                return True
            else:
                return False
        elif method == "IRLS":
            # solve for orthogonal regression parameters directly using IRLS
            param_instance = GNC_WelschParams(WelschInfluenceFunc(), self.__sigma,
                                              sigma_limit=self.__sigma_limit, num_sigma_steps=self.__num_sigma_steps)
            optimiser_instance = IRLS(param_instance, model_instance=LineFitOrthog(),
                                      max_niterations=self.__max_niterations,
                                      diff_thres=self.__diff_thres,
                                      messages_file=self.__messages_file,
                                      debug=self.__debug)
            if optimiser_instance.fit(data, weight=weight, scale=scale):
                self.final_line = self.__convert_model(optimiser_instance.final_model)
                self.final_weight = optimiser_instance.final_weight
                if self.__debug:
                    self.debug_line_list = [(model[0], self.__convert_model(model[1]), model[2], model[3]) for model in optimiser_instance.debug_model_list]
                    self.debug_update_weights_time = optimiser_instance.debug_update_weights_time
                    self.debug_weighted_fit_time = optimiser_instance.debug_weighted_fit_time
                    self.debug_total_time = optimiser_instance.debug_total_time
                    self.debug_n_iterations = optimiser_instance.debug_n_iterations

                return True
            else:
                return False
        else:
            print("Illegal method:",method)
            assert(False)
