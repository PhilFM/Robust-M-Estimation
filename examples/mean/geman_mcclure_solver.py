import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.irls import IRLS
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.geman_mcclure_influence_func import GemanMcClureInfluenceFunc

from gncs_robust_mean import RobustMean

def main(test_run:bool, output_folder:str="../../Output"):
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
    #x_min = 0.7
    #x_max = 1.3
    x_min = x_max = None

    data = np.array([[0.0], # good data
                     [0.4]]) # bad data
    weight = np.array([5.0, # good data
                       4.9]) # bad data

    model_instance = RobustMean()
    influence_func_instance = GemanMcClureInfluenceFunc(sigma=sigma_base)
    param_instance = GNC_WelschParams(influence_func_instance, sigma_base, sigma_limit, num_sigma_steps)
    irls_instance = IRLS(param_instance, model_instance, data, weight=weight, max_niterations=max_niterations, print_warnings=False)
    if irls_instance.run():
        m = irls_instance.final_model
        if not test_run:
            print("IRLS result: m=", m)

    sup_gn_instance = SupGaussNewton(param_instance, model_instance, data, weight=weight, max_niterations=max_niterations, print_warnings=False)
    if sup_gn_instance.run():
        m = sup_gn_instance.final_model
        if not test_run:
            print("Supervised Gauss-Newton result: m=", m)

    # check derivatives
    #residual = np.array([0.01])
    #rhop, Bterm = sup_gn_instance.calc_influence_func_derivatives(residual, 1.0) # scale
    #sup_gn_instance.numeric_derivs_model = True
    #rhopn, Btermn = sup_gn_instance.calc_influence_func_derivatives(residual, 1.0) # scale
    #if not test_run:
    #    print("rhop=",rhop, rhopn, "Bterm=",Bterm,Btermn)

    # check result when scale is included
    scale = np.array([1.0, # good data
                      1.0]) # bad data

    irls_instance = IRLS(param_instance, model_instance, data, weight=weight, scale=scale, max_niterations=max_niterations)
    if irls_instance.run():
        mscale = irls_instance.final_model
        if not test_run:
            print("Scale result difference=", mscale-m)

    # get min and max of data
    y_min = y_max = 0.0

    if x_min is None:
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

    def objective_func(m):
        return sup_gn_instance.objective_func([m])

    def gradient(m):
        a,AlB = sup_gn_instance.weighted_derivs([m],1.0) # lambda_val
        return a[0]

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
    plt.savefig(os.path.join(output_folder, "geman_mcclure_mean.png"), bbox_inches='tight')
    if not test_run:
        plt.show()

    if test_run:
        print("geman_mcclure_solver OK")

if __name__ == "__main__":
    main(False) # test_run
