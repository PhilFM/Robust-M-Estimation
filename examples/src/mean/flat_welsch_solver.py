import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    import sys
    sys.path.append("../../../pypi_package/src")

from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.gnc_null_params import GNC_NullParams
from gnc_smoothie_philfm.linear_model.linear_regressor import LinearRegressor

from flat_welsch_mean import flat_welsch_mean

def main(test_run:bool, output_folder:str="../../../output"):
    # configuration
    showSolution = True
    showGradient = False

    # sigma  0.2 - good
    #        0.8 - too large
    #       0.02 - too small
    sigma = 0.2

    # override x limit 
    #x_min = 0.7
    #x_max = 1.3
    x_min = x_max = None

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

    m = flat_welsch_mean(data, sigma, weight=weight, print_warnings=False, test_run=test_run)
    if not test_run:
        print("Flat Welsch mean result: m=", m)

    # check result when scale is included
    scale = np.array([1.0, # good data
                      1.0]) # bad data
    mscale = flat_welsch_mean(data, sigma, weight=weight, scale=scale, test_run=test_run)
    if not test_run:
        print("Scale result difference=", mscale-m)

    # get min and max of data
    y_min = y_max = 0.0

    if x_min is None:
        if showGradient:
            x_min = -2.0*sigma
            x_max =  2.0*sigma
        else:
            dmin = dmax = data[0]
            for d in data:
                dmin = min(dmin, d)
                dmax = max(dmax, d)
                if not test_run:
                    print("d=", d, " min/max=", dmin, dmax)

            # allow border
            drange = dmax-dmin
            x_min = dmin - 0.05*drange
            x_max = dmax + 0.05*drange

    if not test_run:
        print("x_min=", x_min, " x_max=", x_max)

    mlist = np.linspace(x_min, x_max, num=300)

    # plot stuff
    optimiser_instance = SupGaussNewton(GNC_NullParams(WelschInfluenceFunc(sigma=sigma)), data,
                                        model_instance=LinearRegressor(data[0]), weight=weight)

    def objective_func(m):
        return optimiser_instance.objective_func([m])

    def gradient_func(m):
        a,AlB = optimiser_instance.weighted_derivs([m],1.0) # lambda_b
        return a[0]

    if showGradient:
        for mx in mlist:
            y_min = min(y_min, gradient_func(mx))
            y_max = max(y_max, gradient_func(mx))
    else:
        for mx in mlist:
            y_max = max(y_max, objective_func(mx))

    if not test_run:
        print("y_min=", y_min, " y_max=", y_max)

    y_min *= 1.01 # allow for a small border
    y_max *= 1.01 # allow for a small border

    plt.close("all")
    plt.figure(num=1, dpi=240)
    ax = plt.gca()
    #plt.box(False)
    ax.set_ylim((y_min, y_max))

    hmfv = np.vectorize(objective_func)
    plt.plot(mlist, hmfv(mlist), lw = 1.0)
    for d,w in zip(data,weight, strict=True):
        if d >= x_min and d <= x_max:
            plt.axvline(x = d, color = 'b', ymax = 0.1*w, lw = 1.0)

    if showSolution:
        plt.axvline(x = m[0], color = 'r', label = 'solution', lw = 1.0)

    plt.legend()
    plt.savefig(os.path.join(output_folder, "flat_welsch_solver.png"), bbox_inches='tight')
    if not test_run:
        plt.show()

    if test_run:
        print("flat_welsch_solver OK")

if __name__ == "__main__":
    main(False) # test_run
