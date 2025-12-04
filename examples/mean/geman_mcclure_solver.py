import numpy as np
import matplotlib.pyplot as plt
import os

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.irls import IRLS
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.geman_mcclure_influence_func import GemanMcClureInfluenceFunc

from gncs_robust_mean import RobustMean

def main(testrun:bool, output_folder:str="../../Output"):
    # configuration
    showSolution = True

    # sigma  0.2 - good
    #        0.8 - too large
    #       0.02 - too small
    sigma_base = 0.4

    sigma_limit = 500.0
    num_sigma_steps = 100
    max_niterations = 200

    # override x limit 
    #xMin = 0.7
    #xMax = 1.3
    xMin = xMax = None

    data = np.array([[0.0], # good data
                     [0.4]]) # bad data
    weight = np.array([5.0, # good data
                       4.9]) # bad data

    model_instance = RobustMean()
    influence_func_instance = GemanMcClureInfluenceFunc(sigma=sigma_base)
    param_instance = GNC_WelschParams(influence_func_instance, sigma_base, sigma_limit, num_sigma_steps, max_niterations)
    m = IRLS(param_instance, model_instance, data, weight=weight, max_niterations=max_niterations, print_warnings=False).run()
    if not testrun:
        print("IRLS result: m=", m)

    optimiser_instance = SupGaussNewton(param_instance, model_instance, data, weight=weight, max_niterations=max_niterations, print_warnings=False)
    m = optimiser_instance.run()
    if not testrun:
        print("Supervised Gauss-Newton result: m=", m)

    # check derivatives
    #residual = np.array([0.01])
    #rhop, Bterm = optimiser_instance.calc_influence_func_derivatives(residual, 1.0) # scale
    #optimiser_instance.numeric_derivs_model = True
    #rhopn, Btermn = optimiser_instance.calc_influence_func_derivatives(residual, 1.0) # scale
    #if not testrun:
    #    print("rhop=",rhop, rhopn, "Bterm=",Bterm,Btermn)

    # check result when scale is included
    scale = np.array([1.0, # good data
                      1.0]) # bad data

    mscale = IRLS(param_instance, model_instance, data, weight=weight, scale=scale, max_niterations=max_niterations).run()
    if not testrun:
        print("Scale result difference=", mscale-m)

    # get min and max of data
    yMin = yMax = 0.0

    if xMin is None:
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
        return optimiser_instance.objective_func([m])

    def gradient(m):
        a,AlB = optimiser_instance.weighted_derivs([m],1.0) # lambda_val
        return a[0]

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
    plt.savefig(os.path.join(output_folder, "geman_mcclure_mean.png"), bbox_inches='tight')
    if not testrun:
        plt.show()

    if testrun:
        print("geman_mcclure_solver OK")

if __name__ == "__main__":
    main(False) # testrun
