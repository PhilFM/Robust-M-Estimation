import numpy as np
import math
import os
import json
from pathlib import Path

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

from line_fit_method_unit_error import LineFitMethodUnitError
from line_fit_method_min_deriv  import LineFitMethodMinDeriv
from line_fit_data_model_uniform import LineFitDataModelUniform
from line_fit_data_model_normal  import LineFitDataModelNormal

def randomM11() -> float:
    return 2.0*(np.random.rand()-0.5)

sqrt_2 = math.sqrt(2.0)
inv_sqrt_2 = 1.0/sqrt_2
sqrt_pi = math.sqrt(math.pi)


sup_gn_q_list = np.linspace(0.1, 1.0, 10)

if False: #debug:
    theta_centre = 2.7372292427317007 # 0.5*math.pi
    alpha_centre = 0.20489033552457142 # 0.505
    beta_centre  = 0.4171359771111619
    gamma_centre = 2.66800562663619
    delta_centre = -0.3

    theta_half_range = 0.01
    alpha_half_range = 0.01
    beta_half_range = 0.01
    gamma_half_range = 0.01
    delta_half_range = 0.01

    n_theta_values = 3
    n_alpha_divisions = 3
    n_beta_divisions  = 3
    n_gamma_divisions = 3
    n_delta_values = 3

    n_iterations = 3
    n_subdivisions = 3
    range_scale_factor = 1.0
    f_val_thres = 1.0e-5 # used to indicate successful convergence
else:
    theta_centre = 0.5*math.pi
    delta_centre = 0.0

    theta_half_range = 0.5*math.pi 
    delta_half_range = 1.0 #0.5*math.pi

    n_theta_values = 21
    n_delta_values = 21

    f_val_thres = 1.0e-5 # used to indicate successful convergence
    
def calculate_breakdown_point(method, debug:bool=False, test_run:bool=False, output_folder:str=None):
    best_theta = None
    best_sol = None
    best_delta = None
    best_deriv_cost = 1.e10
    
    #print("n_divisions=",n_divisions)
    theta_dict = {}
    for theta in np.linspace(theta_centre-theta_half_range, theta_centre+theta_half_range, n_theta_values):
        print("theta=",theta,"method=",method.name(),"data_model=",method.data_model().name(),"q=",method.data_model().sup_gn_q())
        theta_best_sol = None
        theta_best_delta = None
        theta_best_deriv_cost = 1.e10
        for delta in np.linspace(delta_centre-delta_half_range, delta_centre+delta_half_range, n_delta_values):
            best_sample,best_val = method.apply(theta, delta)
            if best_sample is not None:
                if theta_best_sol is None or (best_sample[0] < theta_best_sol[0] or (best_sample[0] == theta_best_sol[0] and best_val < theta_best_deriv_cost)):
                    theta_best_deriv_cost = best_val
                    theta_best_delta = delta
                    theta_best_sol = best_sample

        this_theta_dict = {}
        if theta_best_sol is not None:
            this_theta_dict["alpha"] = theta_best_sol[0]
            this_theta_dict["beta"] = 1.0/theta_best_sol[1]
            this_theta_dict["gamma"] = theta_best_sol[1]
            this_theta_dict["delta"] = theta_best_delta
            this_theta_dict["f_val"] = method.data_model().F_tot(1.0/theta_best_sol[1], theta, theta_best_sol[0], theta_best_sol[1], theta_best_delta)

        theta_dict["theta="+str(theta)] = this_theta_dict

        if best_sol is None or (theta_best_sol[0] < best_sol[0] or (theta_best_sol[0] == best_sol[0] and theta_best_deriv_cost < best_deriv_cost)):
            best_theta = theta
            best_deriv_cost = theta_best_deriv_cost
            best_delta = theta_best_delta
            best_sol = theta_best_sol

    return best_theta,best_sol,best_delta,theta_dict,method.data_model().F_tot(1.0/best_sol[1], best_theta, best_sol[0], best_sol[1], best_delta)

def main(test_run:bool, output_folder:str="../../../output"):
    output_folder += "/line_fit/breakdown_point"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    output_dict = {}
    for sup_gn_q in sup_gn_q_list:
        data_models = [LineFitDataModelUniform(sup_gn_q), LineFitDataModelNormal(sup_gn_q)]
        data_model_dict = {}
        for data_model in data_models:
            methods = [LineFitMethodUnitError(data_model), LineFitMethodMinDeriv(data_model)]
            method_dict = {}
            for method in methods:
                theta,sol,delta,theta_dict,f_val = calculate_breakdown_point(method, debug=True, test_run=test_run, output_folder=output_folder)
                #print("theta_dict=",theta_dict)
                this_dict = {}
                this_dict["theta_vals"] = theta_dict
                this_dict["theta"] = theta
                this_dict["alpha"] = sol[0]
                this_dict["beta"]  = 1.0/sol[1]
                this_dict["gamma"] = sol[1]
                this_dict["delta"] = delta
                this_dict["f_val"] = f_val
                method_dict[method.name()] = this_dict

            data_model_dict[data_model.name()] = method_dict

        output_dict["q="+str(sup_gn_q)] = data_model_dict

    with open(os.path.join(output_folder, "output.json"), 'w', encoding='utf-8') as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main(True) # test_run
