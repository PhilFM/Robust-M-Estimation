import numpy as np
import matplotlib.pyplot as plt
import argparse

from gnc_smoothie_philfm.irls import IRLS
from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.null_params import NullParams
from gnc_smoothie_philfm.gnc_irls_p_influence_func import GNC_IRLSpInfluenceFunc
from gnc_smoothie_philfm.gnc_irls_p_params import GNC_IRLSpParams

from gncs_robust_mean import RobustMean

def main(testrun:bool):
    # configuration
    showSolution = True

    # sigma  0.2 - good
    #        0.8 - too large
    #       0.02 - too small
    p = 0.0
    rscale = 0.8
    epsilon_base = 0.2
    epsilon_limit = 1.0
    beta = 0.95

    # override x limit 
    #xMin = 0.7
    #xMax = 1.3
    xMin = xMax = None

    data = np.array([[0.88], [0.93], [1.0], [1.06], [1.1], # good data
                     [10.0], [2.0], [2.5], [3.2]]) # bad data
    weight = np.array([1.0, 1.0, 1.0, 1.0, 1.0, # good data
                       1.0, 1.0, 1.0, 1.0]) # bad data

    model_instance = RobustMean()
    influence_func_instance = GNC_IRLSpInfluenceFunc()
    param_instance = GNC_IRLSpParams(influence_func_instance, p, rscale, epsilon_base, epsilon_limit, beta)
    irlsInstance = IRLS(param_instance, model_instance, data, weight, print_warnings=False)
    m = irlsInstance.run()
    if not testrun:
        print("IRLS Result: m=", m)

    # for checkout derivatives
    #print("rhop=",irlsInstance.updated_weight(np.array([2]),1.0,1.e-5))
    #irlsInstance.numeric_derivs_model = True
    #print("rhopn=",irlsInstance.updated_weight(np.array([2]),1.0,1.e-5))

    # for checking threshold handling
    #print("epsilon_base=",epsilon_base,"rscale=",rscale)
    #threshold = epsilon_base/rscale
    #print("threshold r=",threshold)
    #print("Just before: ",optimiser_instance.param_instance.influence_func_instance.rho(np.array([(threshold-0.00001)**2.0]), 1.0))
    #print("Just after: ",optimiser_instance.param_instance.influence_func_instance.rho(np.array([(threshold+0.00001)**2.0]), 1.0))
                                                                                    
    # for graph plotting
    optimiser_instance = SupGaussNewton(param_instance, model_instance, data, weight=weight, print_warnings=True)

    # Check supervised G-N algorithm
    #m = optimiser_instance.run()
    #print("Result: m=", m)

    # check derivatives
    for r in (0.01, 0.1, 0.5, 2.0):
        residual = np.array([r])
        rhop, Bterm = optimiser_instance.calc_influence_func_derivatives(residual, 1.0) # scale
        optimiser_instance.numeric_derivs_model = True
        rhopn, Btermn = optimiser_instance.calc_influence_func_derivatives(residual, 1.0) # scale
        if not testrun:
            print("rhop=",rhop, rhopn, "Bterm=",Bterm,Btermn)

    # get min and max of data
    yMin = yMax = 0.0

    if xMin == None:
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

    hmfv = np.vectorize(objective_func)
    plt.plot(mlist, hmfv(mlist), lw = 1.0)
    for d,w in zip(data,weight, strict=True):
        if d >= xMin and d <= xMax:
            plt.axvline(x = d, color = 'b', ymax = 0.1*w, lw = 1.0)

    if showSolution:
        plt.axvline(x = m[0], color = 'r', label = 'solution', lw = 1.0)

    plt.legend()
    plt.savefig('../../Output/gnc_irls_p_mean.png', bbox_inches='tight')
    if not testrun:
        plt.show()

    if testrun:
        print("OK")

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--testrun', action="store_true", default=False)
args = parser.parse_args()
main(args.testrun)
