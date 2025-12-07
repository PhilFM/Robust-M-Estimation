import math
import numpy as np
import matplotlib.pyplot as plt

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.null_params import NullParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie_philfm.draw_functions import gncs_draw_data_points

from gncs_robust_mean import RobustMean

def objective_func(m, optimiser_instance) -> float:
    return optimiser_instance.objective_func([m])

def plotResult(optimiser_instance, data, weight, m, sigma, label, mgt, testrun:bool) -> None:
    dmin = dmax = data[0]
    for d in data:
        dmin = min(dmin, d)
        dmax = max(dmax, d)
        #print("d=", d[1], " min/max=", dmin, dmax)

        # allow border
        drange = dmax-dmin
        xMin = dmin - 0.05*drange
        xMax = dmax + 0.05*drange

    mlist = np.linspace(xMin, xMax, num=300)

    plt.figure(num=1, dpi=240)
    gncs_draw_data_points(plt, data, weight, xMin, xMax, len(data), scale=0.05)
    if mgt is not None:
        plt.axvline(x = mgt, color = 'gray', label = 'Ground truth', lw = 1.0, linestyle = 'solid')

    rmfv = np.vectorize(objective_func, excluded="optimiser_instance")
    plt.plot(mlist, rmfv(mlist, optimiser_instance=optimiser_instance), color = 'green', lw = 1.0)
    plt.axvline(x = m,   color = 'green',   label = label,   lw = 1.0, linestyle = 'solid')

    plt.legend()
    plt.savefig("../../../Output/flat_welsch_mean.png", bbox_inches='tight')
    if not testrun:
        plt.show()

def merge(intervals) -> [[float,float]]:
    intervals.sort(key=lambda x: x[0])
    merged = []

    #print("sorted intervals: ", intervals)
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    
    return merged

def mergeOverlap(arr) -> [[float,float]]:
    
    # Sort intervals based on start values
    arr.sort()

    res = []
    res.append(arr[0])

    for i in range(1, len(arr)):
        last = res[-1]
        curr = arr[i]

        # If current interval overlaps with the last merged
        # interval, merge them 
        if curr[0] <= last[1]:
            last[1] = max(last[1], curr[1])
        else:
            res.append(curr)

    return res

def flat_welsch_mean(data, sigma, weight=None, scale=None,
                     max_niterations=50, residual_tolerance=1.e-8, diff_thres=1.e-10, print_warnings=False, mgt=None, testrun:bool=False) -> float:
    # build +/- sigma intervals around data points
    intervals = [] #np.zeros((0,2))
    for d in data:
        intervals.append([d[0]-sigma,d[0]+sigma])

    # build samples separated by sigma covering merged intervals
    mergedIntervals = merge(intervals)
    #print("mergedIntervals=",mergedIntervals)
    sampleX = None
    for interval in mergedIntervals:
        #print("interval:",interval)
        harr = np.linspace(interval[0], interval[1], 1+int((interval[1]-interval[0])/sigma))
        #print("linspace", np.shape(h), h)
        if sampleX is None:
            sampleX = harr
        else:
            sampleX = np.concatenate((sampleX, harr))

    #print("sampleX=",sampleX)

    # get maximum value over samples separated by sigma
    maxVal = 0.0
    maxX = 0.0
    sampleVal = []
    param_instance = NullParams(WelschInfluenceFunc(sigma=sigma))
    optimiser_instance = SupGaussNewton(param_instance, RobustMean(), data, weight, scale)
    for x in sampleX:
        v = optimiser_instance.objective_func([x])
        sampleVal.append(v)
        if v > maxVal:
            maxVal = v
            maxX = x

    if maxVal <= 0.0:
        return 0.0

    # find candidate maxima greater than maxVal*e^-1/2
    thres = math.exp(-0.5)*maxVal
    testValsX = [maxX]
    for i,x in enumerate(sampleX):
        if sampleVal[i] > thres and abs(x-maxX) >= sigma:
            if i == 0 or i == len(sampleX)-1 or (sampleVal[i] > sampleVal[i-1] and sampleVal[i] > sampleVal[i+1]):
                testValsX.append(x)

    if print_warnings:
        print("testValsX=",testValsX)

    # optimise each candidate
    for x in testValsX:
        if print_warnings:
            print("Init m=",x)
            print("data=",data)
            print("weight=",weight)
            print("scale=",scale)
            plotResult(optimiser_instance, data, weight, x, sigma, "Init m", mgt, testrun)

        sup_gn_instance = SupGaussNewton(param_instance, RobustMean(), data, weight=weight, scale=scale,
                                         max_niterations=max_niterations, residual_tolerance=residual_tolerance,
                                         lambda_start=0.99, lambda_max=0.99, diff_thres=diff_thres,
                                         model_start=[x], print_warnings=print_warnings)
        if sup_gn_instance.run():
            m = sup_gn_instance.final_model
            if print_warnings:
                print("sigma=",sigma," m=",m)

            testVal = optimiser_instance.objective_func([m])
            if testVal > maxVal:
                maxVal = testVal
                maxX = m

            if print_warnings:
                print("testVal=",testVal," maxVal=",maxVal," maxX=",maxX)
                plotResult(optimiser_instance, data, weight, m, sigma, "New m", mgt, testrun)

    return maxX
