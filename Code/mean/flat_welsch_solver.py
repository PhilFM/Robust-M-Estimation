import numpy as np
import matplotlib.pyplot as plt
import os

from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.null_params import NullParams

from flat_welsch_mean import flat_welsch_mean
from gncs_robust_mean import RobustMean

def main(testrun:bool, output_folder:str="../../Output"):
    # configuration
    showSolution = True
    showGradient = False

    # sigma  0.2 - good
    #        0.8 - too large
    #       0.02 - too small
    sigma = 0.2

    # override x limit 
    #xMin = 0.7
    #xMax = 1.3
    xMin = xMax = None

    if showGradient:
        data = np.array([[0.0], # good data
                         [0.25], [0.1], [-0.2], [-0.3]]) # bad data
        weight = np.array([5.0, # good data
                           1.0, 1.0, 1.0, 1.0]) # bad data
    else:
        data = np.array([[0.0], # good data
                         [0.4]]) # bad data
        weight = np.array([5.0, # good data
                           4.9]) # bad data

    m = flat_welsch_mean(data, sigma, weight=weight, print_warnings=False, testrun=testrun)
    if not testrun:
        print("Flat Welsch mean result: m=", m)

    # check result when scale is included
    scale = np.array([1.0, # good data
                      1.0]) # bad data
    mscale = flat_welsch_mean(data, sigma, weight=weight, scale=scale, testrun=testrun)
    if not testrun:
        print("Scale result difference=", mscale-m)

    # get min and max of data
    yMin = yMax = 0.0

    if xMin == None:
        if showGradient:
            xMin = -2.0*sigma
            xMax =  2.0*sigma
        else:
            dmin = dmax = data[0]
            for d in data:
                dmin = min(dmin, d)
                dmax = max(dmax, d)
                if not testrun:
                    print("d=", d, " min/max=", dmin, dmax)

            # allow border
            drange = dmax-dmin
            xMin = dmin - 0.05*drange
            xMax = dmax + 0.05*drange

    if not testrun:
        print("xMin=", xMin, " xMax=", xMax)

    mlist = np.linspace(xMin, xMax, num=300)

    # plot stuff
    optimiser_instance = SupGaussNewton(NullParams(WelschInfluenceFunc(sigma=sigma)), RobustMean(), data, weight)

    def objective_func(m):
        return optimiser_instance.objective_func([m])

    if showGradient:
        for mx in mlist:
            #print("x=", mx, " grad=", robustMeanGradient(mx, sigma, data))
            yMin = min(yMin, gradient_func(mx))
            yMax = max(yMax, gradient_func(mx))
    else:
        for mx in mlist:
            yMax = max(yMax, objective_func(mx))

    if not testrun:
        print("yMin=", yMin, " yMax=", yMax)

    yMin *= 1.01 # allow for a small border
    yMax *= 1.01 # allow for a small border

    plt.close("all")
    plt.figure(num=1, dpi=240)
    ax = plt.gca()
    #plt.box(False)
    ax.set_ylim((yMin, yMax))

    hmfv = np.vectorize(objective_func)
    plt.plot(mlist, hmfv(mlist), lw = 1.0)
    for d,w in zip(data,weight, strict=True):
        if d >= xMin and d <= xMax:
            plt.axvline(x = d, color = 'b', ymax = 0.1*w, lw = 1.0)

    if showSolution:
        plt.axvline(x = m[0], color = 'r', label = 'solution', lw = 1.0)

    plt.legend()
    plt.savefig(os.path.join(output_folder, "flat_welsch_solver.png"), bbox_inches='tight')
    if not testrun:
        plt.show()

    if testrun:
        print("flat_welsch_solver OK")

if __name__ == "__main__":
    main(False) # testrun
