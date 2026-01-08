import numpy as np
import matplotlib.pyplot as plt
import os
import time

if __name__ == "__main__":
    import sys
    sys.path.append("../../../pypi_package/src")

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.irls import IRLS
from gnc_smoothie_philfm.gnc_null_params import GNC_NullParams
from gnc_smoothie_philfm.linear_model.linear_regressor import LinearRegressor

# Welsch
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie_philfm.linear_model.linear_regressor_welsch import LinearRegressorWelsch

# Pseudo-Huber
from gnc_smoothie_philfm.pseudo_huber_influence_func import PseudoHuberInfluenceFunc

# Geman-McClure
from gnc_smoothie_philfm.geman_mcclure_influence_func import GemanMcClureInfluenceFunc

# GNC IRLS-p
from gnc_smoothie_philfm.gnc_irls_p_influence_func import GNC_IRLSpInfluenceFunc
from gnc_smoothie_philfm.gnc_irls_p_params import GNC_IRLSpParams

def mean_welsch_solver(data: np.array, x_range: float, use_slow_version: bool, test_run: bool) -> None:
    # estimation parameters
    sigma_base = 1.0
    sigma_limit = x_range
    num_sigma_steps = 20
    max_niterations = 50

    mean_finder = LinearRegressorWelsch(sigma_base, sigma_limit=sigma_limit, num_sigma_steps=num_sigma_steps,
                                        max_niterations=max_niterations, use_slow_version=use_slow_version, print_warnings=False, debug=True)
    if mean_finder.run(data):
        m = mean_finder.final_model[0]
        final_weight = mean_finder.final_weight
        if not test_run:
            print("Welsch Sup-GN optimisation result: m=", m)
            print("  final weights:",final_weight)

        return m
    else:
        return None

def mean_pseudo_huber_solver(data: np.array, x_range: float, use_slow_version: bool, test_run: bool) -> None:
    model_instance = LinearRegressor(data[0])
    influence_func_instance = PseudoHuberInfluenceFunc(sigma=1.0)
    param_instance = GNC_NullParams(influence_func_instance)
    irls_instance = IRLS(param_instance, data, model_instance=model_instance, max_niterations=200, print_warnings=False, debug=True)
    if irls_instance.run():
        m = irls_instance.final_model[0]
        final_weight = irls_instance.final_weight
        if not test_run:
            print("Pseudo-Huber result: m=", m)
            print("  final_weight=",final_weight)

    # check IRLS with scale
    irls_instance = IRLS(GNC_NullParams(influence_func_instance), data, model_instance=model_instance)
    if irls_instance.run():
        mscale = irls_instance.final_model[0]
        if not test_run:
            print("Pseudo-Huber scale result difference=", mscale-m)

def mean_geman_mcclure_solver(data: np.array, x_range: float, test_run: bool, output_folder: str) -> None:
    model_instance = LinearRegressor(data[0])
    p = 0.3
    sigma_base = 1.0/p
    sigma_limit = x_range
    num_sigma_steps = 20
    influence_func_instance = GemanMcClureInfluenceFunc(sigma=sigma_base)
    param_instance = GNC_WelschParams(influence_func_instance, sigma_base, sigma_limit, num_sigma_steps)
    irls_instance = IRLS(param_instance, data, model_instance=model_instance, print_warnings=False)
    if irls_instance.run():
        m = irls_instance.final_model[0]
        if not test_run:
            print("Geman-McClure IRLS result: m=", m)

    sup_gn_instance = SupGaussNewton(param_instance, data, model_instance=model_instance, print_warnings=False, debug=True)
    if sup_gn_instance.run():
        m = sup_gn_instance.final_model[0]
        final_weight = irls_instance.final_weight
        if not test_run:
            print("Geman-McClure Sup-GN result: m=", m)
            print("  final_weight=",final_weight)

def mean_gnc_irls_p_solver(data: np.array, x_range: float, test_run: bool, output_folder: str) -> None:
    p = 0.0
    rscale = 0.8
    epsilon_base = 1.0/0.6667
    epsilon_limit = 1.0
    beta = 0.95

    model_instance = LinearRegressor(data[0])
    influence_func_instance = GNC_IRLSpInfluenceFunc()
    param_instance = GNC_IRLSpParams(influence_func_instance, p, rscale, epsilon_base, epsilon_limit, beta)
    irls_instance = IRLS(param_instance, data, model_instance=model_instance, print_warnings=False, debug=True)
    if irls_instance.run():
        m = irls_instance.final_model[0]
        final_weight = irls_instance.final_weight
        if not test_run:
            print("GNC IRLS-p IRLS Result: m=", m)
            print("  final_weight=",final_weight)

def main(test_run:bool, output_folder:str="../../../output", quick_run:bool=False):
    np.random.seed(0) # We want the numbers to be the same on each run

    # data generation
    n_points = 50 if quick_run else 1000
    x_range = 10.0
    n_tests = 20
    data = []
    for idx in range(n_tests):
        this_data = np.zeros((n_points,1))
        for i in range(n_points):
            this_data[i][0] = np.random.rand()*x_range

        data.append(this_data)

    result_list = []
    start_time = time.time()
    for idx in range(n_tests):
        result_list.append(mean_welsch_solver(data[idx], x_range, False, test_run))

    if not test_run:
        print("Welsch time:", time.time()-start_time)

    result_list = np.array(result_list)
    if not test_run:
        print(result_list)

    slow_result_list = []
    start_time = time.time()
    for idx in range(n_tests):
        slow_result_list.append(mean_welsch_solver(data[idx], x_range, True, test_run))

    if not test_run:
        print("Welsch slow time:", time.time()-start_time)

    slow_result_list = np.array(slow_result_list)

    if not test_run:
        print("Diffs:",result_list-slow_result_list)

    mean_pseudo_huber_solver(data, x_range, test_run, output_folder)
    mean_geman_mcclure_solver(data, x_range, test_run, output_folder)
    mean_gnc_irls_p_solver(data, x_range, test_run, output_folder)

    if test_run:
        print("mean_solve_speed OK")

if __name__ == "__main__":
    main(False) # test_run
