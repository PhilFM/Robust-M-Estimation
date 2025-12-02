import numpy as np
import matplotlib.pyplot as plt
import argparse

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.irls import IRLS
from gnc_smoothie_philfm.draw_functions import gncs_draw_data_points
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.null_params import NullParams
from gnc_smoothie_philfm.gnc_irls_p_params import GNC_IRLSpParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie_philfm.pseudo_huber_influence_func import PseudoHuberInfluenceFunc
from gnc_smoothie_philfm.gnc_irls_p_influence_func import GNC_IRLSpInfluenceFunc
from gnc_smoothie_philfm.plt_alg_vis import gncs_draw_vline, gncs_draw_curve

from flat_welsch_mean import flat_welsch_mean
from gncs_robust_mean import RobustMean

N = 10
xrange = 10.0
sigma_base = 0.5
sigma_limit = xrange
num_sigma_steps = 20 #50
max_niterations = 100
diff_thres = 1.0e-15

def objective_func(m, optimiser_instance):
    return optimiser_instance.base.objective_func([m])

def plotResult(data, weight,
               gncWelschOptimiserInstance, mgncw, msupgnw, mflat,
               pseudoHuberOptimiserInstance, mhuber, mirlshuber,
               gncIrlspOptimiserInstance, mgncirlsp,
               plot_count:int,
               testrun:bool):
    dmin = dmax = data[0][0]
    for d in data:
        dmin = min(dmin, d[0])
        dmax = max(dmax, d[0])
        #print("d=", d[1], " min/max=", dmin, dmax)

    # allow border
    drange = dmax-dmin
    xMin = dmin - 0.05*drange
    xMax = dmax + 0.05*drange

    mlist = np.linspace(xMin, xMax, num=300)

    plt.figure(num=plot_count, dpi=240)
    ax = plt.gca()
    rmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
    key = ("Flat", "Welsch", "GNC_Welsch")
    gncs_draw_curve(plt, rmfv(mlist, optimiser_instance=gncWelschOptimiserInstance), key, xvalues=mlist, drawMarkers=False, hlightXValue=mflat, ax=ax)
    gncs_draw_vline(plt, mflat,       key, useLabel=False)
    key = ("SupGN", "Welsch", "GNC_Welsch")
    gncs_draw_curve(plt, rmfv(mlist, optimiser_instance=gncWelschOptimiserInstance), key, xvalues=mlist, drawMarkers=False, hlightXValue=msupgnw, ax=ax)
    gncs_draw_vline(plt, msupgnw,     key, useLabel=False)
    gncs_draw_vline(plt, mgncw, ("IRLS", "Welsch",      "GNC_Welsch"))
    key = ("SupGN", "PseudoHuber", "Welsch")
    gncs_draw_curve(plt, 0.03*rmfv(mlist, optimiser_instance=pseudoHuberOptimiserInstance), key, xvalues=mlist, drawMarkers=False, hlightXValue=mhuber, ax=ax)
    gncs_draw_vline(plt, mhuber,      key, useLabel=False)
    #gncs_draw_vline(plt, mirlshuber, ("IRLS", "PseudoHuber", "Welsch"))
    key = ("IRLS", "GNC_IRLSp", "GNC_IRLSp0")
    gncs_draw_curve(plt, 0.02*rmfv(mlist, optimiser_instance=gncIrlspOptimiserInstance), key, xvalues=mlist, drawMarkers=False, hlightXValue=mgncirlsp, ax=ax)
    gncs_draw_vline(plt, mgncirlsp,   key, useLabel=False)

    gncs_draw_data_points(plt, data, weight, xMin, xMax, N)
    plt.legend()
    plt.savefig("../../Output/irlsCheck.png", bbox_inches='tight')
    if not testrun:
        plt.show()
    
def main(testrun:bool):
    np.random.seed(0) # We want the numbers to be the same on each run

    plot_count = 1
    for test_idx in range(0,10):
        data = np.zeros((N,1))
        weight = np.zeros(N)
        if test_idx == 0:
            for j in range(N):
                weight[j] = 1.0
                if j == 0:
                    data[j] = [0.89]
                elif j < 3:
                    data[j] = [0.9 + 0.01*j]
                else:
                    data[j] = [j*xrange/N]
        else:
            for j in range(N):
                d = np.random.rand()*xrange
                weight[j] = 1.0
                data[j] = [d]

        print_warnings = False #True if test_idx == 0 else False

        model_instance = RobustMean()

        #print("data(1)=",data)
        param_instance = GNC_WelschParams(WelschInfluenceFunc(), sigma_base, sigma_limit, num_sigma_steps)
        mgncw = IRLS(param_instance, model_instance, data, weight=weight,
                     max_niterations=max_niterations, diff_thres=diff_thres, print_warnings=print_warnings).run()
        gncWelschOptimiserInstance = SupGaussNewton(param_instance, model_instance, data, weight=weight,
                                                    max_niterations=max_niterations, diff_thres=diff_thres, print_warnings=print_warnings)
        msupgnw = gncWelschOptimiserInstance.run()

        mflat = flat_welsch_mean(data, sigma_base, weight,
                                 max_niterations=max_niterations, diff_thres=diff_thres, print_warnings=print_warnings)

        param_instance = NullParams(PseudoHuberInfluenceFunc(sigma_base))
        pseudoHuberOptimiserInstance = SupGaussNewton(param_instance, model_instance, data, weight=weight,
                                                      max_niterations=max_niterations, diff_thres=diff_thres, print_warnings=print_warnings)
        mhuber = pseudoHuberOptimiserInstance.run()
        mirlshuber = IRLS(param_instance, model_instance, data, weight=weight,
                          max_niterations=max_niterations, diff_thres=diff_thres, print_warnings=print_warnings).run()

        # GNC IRLS-p params [p,epsilon_base,epsilon_limit,rscale,beta]
        param_instance = GNC_IRLSpParams(GNC_IRLSpInfluenceFunc(), 0.0, 0.01, 1.0, 1.0/xrange, 0.8)
        mgncirlsp = IRLS(param_instance, model_instance, data, weight=weight,
                         max_niterations=max_niterations, diff_thres=diff_thres, print_warnings=print_warnings).run()
        gncIrlspOptimiserInstance = SupGaussNewton(param_instance, model_instance, data, weight=weight,
                                                   max_niterations=max_niterations, diff_thres=diff_thres, print_warnings=print_warnings)

        plotResult(data, weight,
                   gncWelschOptimiserInstance, mgncw, msupgnw, mflat,
                   pseudoHuberOptimiserInstance, mhuber, mirlshuber,
                   gncIrlspOptimiserInstance, mgncirlsp,
                   plot_count, testrun)
        plot_count += 1

    if testrun:
        print("OK")

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--testrun', action="store_true", default=False)
args = parser.parse_args()
main(args.testrun)
