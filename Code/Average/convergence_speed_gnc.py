import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

sys.path.append("../Library")
from SupGaussNewton import SupGaussNewton
from IRLS import IRLS
from GNC_WelschParams import GNC_WelschParams
from NullParams import NullParams
from GNC_IRLSpParams import GNC_IRLSpParams
from WelschInfluenceFunc import WelschInfluenceFunc
from PseudoHuberInfluenceFunc import PseudoHuberInfluenceFunc
from GNC_IRLSpInfluenceFunc import GNC_IRLSpInfluenceFunc
from draw_functions import drawDataPoints
import pltAlgVis

from RobustMean import RobustMean

N = 10
xgtrange = 10.0
sigmaPop = 1.0
num_sigma_steps = 50
max_niterations = 100
residual_tolerance = 1.e-8
lambda_start = 1.0
lambda_scale = 2.0 #1.0
diff_thres = 1.0e-12
gradThres = None #1.0e-12
welsch_p = 0.66666667
outlierFraction = 0.5

def objective_func(m, optimiser_instance):
    return optimiser_instance.base.objective_func([m])

def plotResult(data, weight,
               mwelschopt, welschOptimiserInstance,
               mhuber,     pseudoHuberOptimiserInstance,
               mgncirlsp,  gncIrlspOptimiserInstance,
               mgt,
               plot_count:int,
               testrun:bool):
    dmin = dmax = data[0]
    for d in data:
        dmin = min(dmin, d)
        dmax = max(dmax, d)
        #print("d=", d, " min/max=", dmin, dmax)

        # allow border
        drange = dmax-dmin
        xMin = dmin - 0.05*drange
        xMax = dmax + 0.05*drange

    mlist = np.linspace(xMin, xMax, num=300)

    plt.figure(num=plot_count, dpi=240)
    drawDataPoints(plt, data, weight, xMin, xMax, len(data), scale=0.05)
    if mgt is not None:
        pltAlgVis.drawVLine(plt, mgt, ("GroundTruth","",""))

    ax = plt.gca()

    rmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
    key = ("SupGN", "Welsch", "GNC_Welsch")
    pltAlgVis.drawCurve(plt, rmfv(mlist, optimiser_instance=welschOptimiserInstance), key, xvalues=mlist, drawMarkers=False, hlightXValue=mwelschopt, ax=ax)
    pltAlgVis.drawVLine(plt, mwelschopt, key, useLabel=False)

    rmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
    key = ("SupGN", "PseudoHuber", "Welsch")
    pltAlgVis.drawCurve(plt, 0.05*rmfv(mlist, optimiser_instance=pseudoHuberOptimiserInstance), key, xvalues=mlist, drawMarkers=False, hlightXValue=mhuber, ax=ax)
    pltAlgVis.drawVLine(plt, mhuber, key, useLabel=False)

    rmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
    key = ("IRLS", "GNC_IRLSp", "GNC_IRLSp0")
    pltAlgVis.drawCurve(plt, 0.003*rmfv(mlist, optimiser_instance=gncIrlspOptimiserInstance), key, xvalues=mlist, drawMarkers=False, hlightXValue=mgncirlsp, ax=ax)
    pltAlgVis.drawVLine(plt, mgncirlsp, key, useLabel=False)

    plt.legend()
    plt.savefig("../../Output/convergence_speed_gnc.png", bbox_inches='tight')
    if not testrun:
        plt.show()

def main(testrun:bool):
    np.random.seed(0) # We want the numbers to be the same on each run

    plot_count = 1
    for test_idx in range(0,10):
        N0 = int((1.0-outlierFraction)*N+0.5)
        sigmaPop = 1.0
        xgtborder = 3.0*sigmaPop
        mgt = np.random.rand()*xgtrange + xgtborder
        data = np.zeros((N,1))
        weight = np.zeros(N)
        for j in range(N0):
            weight[j] = 1.0
            data[j] = [np.random.normal(loc=mgt, scale=sigmaPop)]

        for j in range(N-N0):
            weight[N0+j] = 1.0
            data[N0+j] = [np.random.rand()*(xgtrange + 2.0*xgtborder)]

        print_warnings = False

        if not testrun:
            print("test_idx=",test_idx,"mgt=",mgt)

        sigma_base = sigmaPop/welsch_p
        sigma_limit = xgtrange

        model_instance = RobustMean()

        welschOptimiserInstance = SupGaussNewton(GNC_WelschParams(WelschInfluenceFunc(), sigma_base, sigma_limit, num_sigma_steps),
                                                 model_instance, data, weight=weight,
                                                 max_niterations=max_niterations, residual_tolerance=residual_tolerance,
                                                 lambda_start=lambda_start, lambda_scale=lambda_scale, diff_thres=diff_thres,
                                                 print_warnings=print_warnings)
        mwelschopt = welschOptimiserInstance.run()
        if not testrun:
            print("mwelschopt-mgt=",mwelschopt-mgt)

        pseudoHuberOptimiserInstance = IRLS(NullParams(PseudoHuberInfluenceFunc(sigma=sigma_base)), model_instance, data, weight=weight,
                                            max_niterations=max_niterations, diff_thres=diff_thres, print_warnings=print_warnings)
        mhuber = pseudoHuberOptimiserInstance.run()
        if not testrun:
            print("mhuber-mgt=",mhuber-mgt)

        pseudoHuberSupGNOptimiserInstance = SupGaussNewton(NullParams(PseudoHuberInfluenceFunc(sigma=sigma_base)), model_instance, data, weight=weight,
                                                           max_niterations=max_niterations, diff_thres=diff_thres, print_warnings=print_warnings)
    
        gncIrlsp_p = 0.0
        gncIrlsp_rscale = 1.0/xgtrange
        gncIrlsp_epsilon_base = gncIrlsp_rscale*sigma_base
        gncIrlsp_epsilon_limit = gncIrlsp_rscale*sigma_limit
        gncIrlsp_beta = math.exp((math.log(sigma_base) - math.log(sigma_limit))/(num_sigma_steps - 1.0))
        gncIrlspParamInstance = GNC_IRLSpParams(GNC_IRLSpInfluenceFunc(),
                                                gncIrlsp_p, gncIrlsp_rscale, gncIrlsp_epsilon_base, gncIrlsp_epsilon_limit, gncIrlsp_beta)
        gncIrlspOptimiserInstance = IRLS(gncIrlspParamInstance, model_instance, data, weight=weight,
                                         max_niterations=max_niterations, diff_thres=diff_thres, print_warnings=print_warnings)
        mgncirlsp = gncIrlspOptimiserInstance.run()
        if not testrun:
            print("mgncirlsp-mgt=",mgncirlsp-mgt)

        gncIrlspSupGNOptimiserInstance = SupGaussNewton(gncIrlspParamInstance, model_instance, data, weight=weight,
                                                        max_niterations=max_niterations, residual_tolerance=residual_tolerance,
                                                        lambda_start=lambda_start, lambda_scale=lambda_scale, diff_thres=diff_thres,
                                                        print_warnings=print_warnings)
        mgncirlspopt = gncIrlspSupGNOptimiserInstance.run()
        if not testrun:
            print("mgncirlspopt-mgt=",mgncirlspopt-mgt)

        plotResult(data, weight,
                   mwelschopt, welschOptimiserInstance,
                   mhuber,     pseudoHuberSupGNOptimiserInstance,
                   mgncirlsp,  gncIrlspSupGNOptimiserInstance,
                   mgt,
                   plot_count,
                   testrun)
        plot_count += 1

    if testrun:
        print("OK")

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--testrun', action="store_true", default=False)
args = parser.parse_args()
main(args.testrun)
