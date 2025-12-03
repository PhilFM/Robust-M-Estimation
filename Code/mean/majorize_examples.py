import numpy as np
import matplotlib.pyplot as plt
import os

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.gnc_irls_p_params import GNC_IRLSpParams
from gnc_smoothie_philfm.null_params import NullParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie_philfm.pseudo_huber_influence_func import PseudoHuberInfluenceFunc
from gnc_smoothie_philfm.geman_mcclure_influence_func import GemanMcClureInfluenceFunc
from gnc_smoothie_philfm.gnc_irls_p_influence_func import GNC_IRLSpInfluenceFunc

from gncs_robust_mean import RobustMean

def objective_func(x, optimiser_instance):
    if optimiser_instance.objective_func_sign() < 0.0:
        return 1.0-optimiser_instance.objective_func([x])
    else:
        return optimiser_instance.objective_func([x])

def gradient(x, optimiser_instance):
    a,AlB = optimiser_instance.weighted_derivs([x], 1.0) # lambda_val
    return optimiser_instance.objective_func_sign()*a[0]

def centredQuadratic(x, a, c):
    return a*x*x + c

def drawMajorizer(plt, mlist, u, rhou, rhopu, colour):
    a = 0.5*rhopu/u
    c = rhou - a*u*u
    rmfv = np.vectorize(centredQuadratic, excluded={"a", "c"})
    plt.plot(mlist, rmfv(mlist, a=a, c=c), color = colour, lw = 1.0, label = "Majorizer u=" + str(u))

def plotResult(optimiser_instance, uValues:list, label:str, output_folder, file_name:str, testrun:bool):
    xMax = 3.0
    mlist = np.linspace(-xMax, xMax, num=300)

    plt.close("all")
    plt.figure(num=1, dpi=240)
    ax = plt.gca()

    colours = ["blue", "purple"]
    for u,col in zip(uValues,colours, strict=True):
        drawMajorizer(plt, mlist, u, objective_func(u,optimiser_instance), gradient(u,optimiser_instance), col)

    rmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
    plt.plot(mlist, rmfv(mlist, optimiser_instance=optimiser_instance), color = 'green', lw=1.0, label=label)

    ax.set_xlabel(r'r')

    plt.legend()
    plt.savefig(os.path.join(output_folder, file_name + ".png"), bbox_inches='tight')
    if not testrun:
        plt.show()

def main(testrun:bool, output_folder:str="../../Output"):
    model_instance = RobustMean()

    p = 0.0
    rscale = 1.0
    epsilon = 0.1
    param_instance = GNC_IRLSpParams(GNC_IRLSpInfluenceFunc(), p, rscale, epsilon)
    plotResult(SupGaussNewton(param_instance, model_instance, [[0.0]], weight=[1.0], numeric_derivs_influence=True),
               [1.0, 2.0], "GNC IRLSp0 influence function", output_folder, "gnc_irls_p0_majorizers", testrun)

    p = 0.5
    param_instance = GNC_IRLSpParams(GNC_IRLSpInfluenceFunc(), p, rscale, epsilon)
    plotResult(SupGaussNewton(param_instance, model_instance, [[0.0]], weight=[1.0], numeric_derivs_influence=True),
               [1.0, 2.0], "GNC IRLSp0.5 influence function", output_folder, "gnc_irls_ph_majorizers", testrun)

    p = 1.0
    param_instance = GNC_IRLSpParams(GNC_IRLSpInfluenceFunc(), p, rscale, epsilon)
    plotResult(SupGaussNewton(param_instance, model_instance, [[0.0]], weight=[1.0], numeric_derivs_influence=True),
               [1.0, 2.0], "GNC IRLSp1 influence function", output_folder, "gnc_irls_p1_majorizers", testrun)

    sigma = 1.0
    param_instance = GNC_WelschParams(WelschInfluenceFunc(), sigma)
    plotResult(SupGaussNewton(param_instance, model_instance, [[0.0]], weight=[1.0]),
               [1.5, 2.0], "Welsch influence function", output_folder, "welsch_majorizers", testrun)

    param_instance = NullParams(PseudoHuberInfluenceFunc(sigma))
    plotResult(SupGaussNewton(param_instance, model_instance, [[0.0]], weight=[1.0]),
               [1.5, 2.0], "Pseudo-Huber influence function", output_folder, "pseudo_huber_majorizers", testrun)
    
    param_instance = NullParams(GemanMcClureInfluenceFunc(sigma))
    plotResult(SupGaussNewton(param_instance, model_instance, [[0.0]], weight=[1.0]),
               [1.0, 2.0], "Geman-McClure influence function", output_folder, "geman_mcclure_majorizers", testrun)

    if testrun:
        print("majorize_examples OK")
