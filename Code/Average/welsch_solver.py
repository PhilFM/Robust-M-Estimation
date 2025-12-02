import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

sys.path.append("../Library")
from SupGaussNewton import SupGaussNewton
from IRLS import IRLS
from GNC_WelschParams import GNC_WelschParams
from WelschInfluenceFunc import WelschInfluenceFunc

from RobustMean import RobustMean

def main(testrun:bool):
    # configuration
    showSolution = True
    showGradient = False

    # sigma  0.2 - good
    #        0.8 - too large
    #       0.02 - too small
    sigma_base = 0.2

    sigma_limit = 500.0
    num_sigma_steps = 100
    max_niterations = 200

    xMin = xMax = None

    # data is a list of [weight, value] pairs
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

    model_instance = RobustMean()
    param_instance = GNC_WelschParams(WelschInfluenceFunc(), sigma_base, sigma_limit=sigma_limit,
                                      num_sigma_steps=num_sigma_steps, max_niterations=max_niterations)
    m = IRLS(param_instance, model_instance, data, weight=weight, max_niterations=max_niterations, print_warnings=False).run()
    if not testrun:
        print("IRLS result: m=", m)

    optimiser_instance = SupGaussNewton(param_instance, model_instance, data, weight=weight, max_niterations=max_niterations, print_warnings=False)
    m = optimiser_instance.run()
    if not testrun:
        print("Supervised Gauss-Newton optimisation result: m=", m)

    # check result when scale is included
    if showGradient:
        scale = np.array([1.0, # good data
                          1.0, 1.0, 1.0, 1.0]) # bad data
    else:
        scale = np.array([1.0, # good data
                          1.0]) # bad data

    mscale = SupGaussNewton(param_instance, model_instance, data, weight=weight, scale=scale, max_niterations=max_niterations).run()
    if not testrun:
        print("Scale result difference=", mscale-m)

    # get min and max of data
    yMin = yMax = 0.0

    if xMin == None:
        if showGradient:
            xMin = -2.0*sigma_base
            xMax =  2.0*sigma_base
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

    def objective_func(m):
        return optimiser_instance.base.objective_func([m])

    def gradient_func(m):
        a,AlB = optimiser_instance.weighted_derivs([m],1.0) # lambda_val
        return a[0]

    if showGradient:
        for mx in mlist:
            yMin = min(yMin, gradient_func(mx))
            yMax = max(yMax, gradient_func(mx))
    else:
        for mx in mlist:
            yMax = max(yMax, objective_func(mx))

    if not testrun:
        print("yMin=", yMin, " yMax=", yMax)

    yMin *= 1.01 # allow for a small border
    yMax *= 1.01 # allow for a small border

    plt.figure(num=1, dpi=240)
    ax = plt.gca()
    #plt.box(False)
    ax.set_ylim((yMin, yMax))

    if showGradient:
        rmgv = np.vectorize(gradient_func)
        plt.plot(mlist, rmgv(mlist), lw = 1.0)
        for d,w in zip(data,weight, strict=True):
            if d >= xMin and d <= xMax:
                plt.axvline(x = d, color = 'b', ymax = 0.05*w, lw = 1.0)

        #fig.gca().set_ylabel(r'$\lambda$')
        plt.axhline(y = 0.0, color = 'b', label = '', lw = 1.0)
        plt.axvline(x = -sigma_base, color = 'g', label = r'x=-$\sigma$', lw = 1.0)
        plt.axvline(x =  sigma_base, color = 'r', label = r'x= $\sigma$', lw = 1.0)
        if showSolution:
            plt.axvline(x = m[0], color = 'r', label = 'solution', lw = 1.0)
    else:
        hmfv = np.vectorize(objective_func)
        plt.plot(mlist, hmfv(mlist), lw = 1.0)
        for d,w in zip(data,weight, strict=True):
            if d >= xMin and d <= xMax:
                plt.axvline(x = d, color = 'b', ymax = 0.1*w, lw = 1.0)

        if showSolution:
            plt.axvline(x = m[0], color = 'r', label = 'solution', lw = 1.0)

    plt.legend()
    plt.savefig('../../Output/welsch_mean.png', bbox_inches='tight')
    if not testrun:
        plt.show()

    if testrun:
        print("OK")

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--testrun', action="store_true", default=False)
args = parser.parse_args()
main(args.testrun)
