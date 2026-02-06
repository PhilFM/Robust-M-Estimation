import math
import numpy as np
import matplotlib.pyplot as plt
import os

from gnc_smoothie.sup_gauss_newton import SupGaussNewton
from gnc_smoothie.gnc_null_params import GNC_NullParams
from gnc_smoothie.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie.draw_functions import gncs_draw_data_points
from gnc_smoothie.linear_model.linear_regressor import LinearRegressor

def objective_func(m, optimiser_instance) -> float:
    return optimiser_instance.objective_func([m])

def plot_result(optimiser_instance, data, weight, m, sigma, label, m_gt, output_folder:str, test_run:bool) -> None:
    dmin = min(data)
    dmax = max(data)

    # allow border
    drange = dmax-dmin
    x_min = dmin - 0.05*drange
    x_max = dmax + 0.05*drange

    mlist = np.linspace(x_min, x_max, num=300)

    plt.figure(num=1, dpi=240)
    gncs_draw_data_points(plt, data, x_min, x_max, len(data), weight=weight, scale=0.05)
    if m_gt is not None:
        plt.axvline(x = m_gt, color = 'gray', label = 'Ground truth', lw = 1.0, linestyle = 'solid')

    rmfv = np.vectorize(objective_func, excluded="optimiser_instance")
    plt.plot(mlist, rmfv(mlist, optimiser_instance=optimiser_instance), color = 'green', lw = 1.0)
    plt.axvline(x = m,   color = 'green',   label = label,   lw = 1.0, linestyle = 'solid')

    plt.legend()
    plt.savefig(os.path.join(output_folder, "flat_welsch_mean.png"), bbox_inches='tight')
    if not test_run:
        plt.show()

def merge(intervals) -> [[float,float]]:
    intervals.sort(key=lambda x: x[0])
    merged = []

    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    
    return merged

def flat_welsch_mean(data, sigma, weight=None, scale=None,
                     max_niterations=100, residual_tolerance=1.e-8, diff_thres=1.e-12, messages_file=None, m_gt=None,
                     output_folder:str="../../../output", test_run:bool=False) -> float:
    # build +/- sigma intervals around data points
    intervals = [] #np.zeros((0,2))
    for d in data:
        intervals.append([d[0]-sigma,d[0]+sigma])

    # build samples separated by sigma covering merged intervals
    mergedIntervals = merge(intervals)
    sample_x = None
    for interval in mergedIntervals:
        harr = np.linspace(interval[0], interval[1], 1+int((interval[1]-interval[0])/sigma))
        if sample_x is None:
            sample_x = harr
        else:
            sample_x = np.concatenate((sample_x, harr))

    # get maximum value over samples separated by sigma
    max_val = 0.0
    max_x = 0.0
    sample_val = []
    param_instance = GNC_NullParams(WelschInfluenceFunc(sigma=sigma))
    optimiser_instance = SupGaussNewton(param_instance, data, model_instance=LinearRegressor(data[0]), weight=weight, scale=scale)
    for x in sample_x:
        v = optimiser_instance.objective_func([x])
        sample_val.append(v)
        if v > max_val:
            max_val = v
            max_x = x

    if max_val <= 0.0:
        return 0.0

    # find candidate maxima greater than max_val*e^-1/2
    thres = math.exp(-0.5)*max_val
    test_vals_x = [max_x]
    for i,x in enumerate(sample_x):
        if sample_val[i] > thres and abs(x-max_x) >= sigma:
            if i == 0 or i == len(sample_x)-1 or (sample_val[i] > sample_val[i-1] and sample_val[i] > sample_val[i+1]):
                test_vals_x.append(x)

    if messages_file is not None:
        print("test_vals_x=",test_vals_x, file=messages_file)

    # optimise each candidate
    for x in test_vals_x:
        if messages_file is not None:
            print("Init m=",x, file=messages_file)
            print("data=",data, file=messages_file)
            print("weight=",weight, file=messages_file)
            print("scale=",scale, file=messages_file)
            plot_result(optimiser_instance, data, weight, x, sigma, "Init m", m_gt, output_folder, test_run)

        sup_gn_instance = SupGaussNewton(param_instance, data, model_instance=LinearRegressor(data[0]), weight=weight, scale=scale,
                                         max_niterations=max_niterations, residual_tolerance=residual_tolerance,
                                         lambda_start=0.99, lambda_max=0.99, diff_thres=diff_thres,
                                         messages_file=messages_file)
        if sup_gn_instance.run(model_start=[x]):
            m = sup_gn_instance.final_model
            testVal = optimiser_instance.objective_func([m])
            if testVal > max_val:
                max_val = testVal
                max_x = m

            if messages_file is not None:
                print("testVal=",testVal," max_val=",max_val," max_x=",max_x, file=messages_file)
                plot_result(optimiser_instance, data, weight, m, sigma, "New m", m_gt, output_folder, test_run)

    if messages_file is not None:
        o_val = optimiser_instance.objective_func([max_x])
        print("flat Welsch mean=",max_x,"objective_func=",o_val, file=messages_file)

    return max_x
