import numpy as np
import matplotlib.pyplot as plt
import sys, os
import argparse

sys.path.append(os.path.join("..", "Library"))
from SupGaussNewton import SupGaussNewton
from GNC_WelschParams import GNC_WelschParams
from GNC_IRLSpParams import GNC_IRLSpParams
from NullParams import NullParams
from WelschInfluenceFunc import WelschInfluenceFunc
from PseudoHuberInfluenceFunc import PseudoHuberInfluenceFunc
from GemanMcClureInfluenceFunc import GemanMcClureInfluenceFunc
from GNC_IRLSpInfluenceFunc import GNC_IRLSpInfluenceFunc

sys.path.append("..")
from RobustMean import RobustMean

def objective_func(x, optimiser_instance):
    if optimiser_instance.base.param_instance.influence_func_instance.objective_func_sign() < 0.0:
        return 1.0-optimiser_instance.base.objective_func([x])
    else:
        return optimiser_instance.base.objective_func([x])

def gradient(x, optimiser_instance):
    a,AlB = optimiser_instance.weighted_derivs([x], 1.0) # lambda_val
    return optimiser_instance.base.param_instance.influence_func_instance.objective_func_sign()*a[0]

def centredQuadratic(x, a, c):
    return a*x*x + c

def drawMajorizer(plt, mlist, u, rhou, rhopu, colour):
    a = 0.5*rhopu/u
    c = rhou - a*u*u
    rmfv = np.vectorize(centredQuadratic, excluded={"a", "c"})
    plt.plot(mlist, rmfv(mlist, a=a, c=c), color = colour, lw = 1.0, label = "Majorizer u=" + str(u))

def plotResult(optimiser_instance, uValues:list, label:str, fileName:str, plot_count, testrun:bool):
    xMax = 3.0
    mlist = np.linspace(-xMax, xMax, num=300)

    plt.figure(num=plot_count, dpi=240)
    ax = plt.gca()

    colours = ["blue", "purple"]
    for u,col in zip(uValues,colours, strict=True):
        drawMajorizer(plt, mlist, u, objective_func(u,optimiser_instance), gradient(u,optimiser_instance), col)

    rmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
    plt.plot(mlist, rmfv(mlist, optimiser_instance=optimiser_instance), color = 'green', lw=1.0, label=label)

    ax.set_xlabel(r'r')

    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(sys.path[0])), "Output", fileName + ".png"), bbox_inches='tight')
    if not testrun:
        plt.show()

def main(testrun:bool):
    model_instance = RobustMean()

    plot_count = 1

    p = 0.0
    rscale = 1.0
    epsilon = 0.1
    param_instance = GNC_IRLSpParams(GNC_IRLSpInfluenceFunc(), p, rscale, epsilon)
    plotResult(SupGaussNewton(param_instance, model_instance, [[0.0]], weight=[1.0], numeric_derivs_influence=True),
               [1.0, 2.0], 'GNC IRLSp0 influence function', "gnc_irls_p0_majorizers", plot_count, testrun)
    plot_count += 1

    p = 0.5
    param_instance = GNC_IRLSpParams(GNC_IRLSpInfluenceFunc(), p, rscale, epsilon)
    plotResult(SupGaussNewton(param_instance, model_instance, [[0.0]], weight=[1.0], numeric_derivs_influence=True),
               [1.0, 2.0], 'GNC IRLSp0.5 influence function', "gnc_irls_ph_majorizers", plot_count, testrun)
    plot_count += 1

    p = 1.0
    param_instance = GNC_IRLSpParams(GNC_IRLSpInfluenceFunc(), p, rscale, epsilon)
    plotResult(SupGaussNewton(param_instance, model_instance, [[0.0]], weight=[1.0], numeric_derivs_influence=True),
               [1.0, 2.0], 'GNC IRLSp1 influence function', "gnc_irls_p1_majorizers", plot_count, testrun)
    plot_count += 1

    sigma = 1.0
    param_instance = GNC_WelschParams(WelschInfluenceFunc(), sigma)
    plotResult(SupGaussNewton(param_instance, model_instance, [[0.0]], weight=[1.0]),
               [1.5, 2.0], 'Welsch influence function', "welsch_majorizers", plot_count, testrun)
    plot_count += 1

    param_instance = NullParams(PseudoHuberInfluenceFunc(sigma))
    plotResult(SupGaussNewton(param_instance, model_instance, [[0.0]], weight=[1.0]),
               [1.5, 2.0], 'Pseudo-Huber influence function', "pseudo_huber_majorizers", plot_count, testrun)
    plot_count += 1
    
    param_instance = NullParams(GemanMcClureInfluenceFunc(sigma))
    plotResult(SupGaussNewton(param_instance, model_instance, [[0.0]], weight=[1.0]),
               [1.0, 2.0], 'Geman-McClure influence function', "geman_mcclure_majorizers", plot_count, testrun)
    plot_count += 1

    if testrun:
        print("OK")

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--testrun', action="store_true", default=False)
args = parser.parse_args()
main(args.testrun)
