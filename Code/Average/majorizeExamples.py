import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("Welsch")
from WelschMean import WelschMean

sys.path.append("PseudoHuber")
from PseudoHuberMean import PseudoHuberMean

sys.path.append("GemanMcClure")
from GemanMcClureMean import GemanMcClureMean

sys.path.append("GNC_IRLSp")
from GNC_IRLSpMean import GNC_IRLSpMean

sys.path.append("GNC_TLS")
from GNC_TLSMean import GNC_TLSMean

sys.path.append("../Library")
from GNC_WelschParams import GNC_WelschParams
from WelschParams import WelschParams
from GNC_IRLSpParams import GNC_IRLSpParams
from GNC_TLSParams import GNC_TLSParams

def objectiveFunc(x, algInstance):
    if algInstance.objectiveFuncSign() < 0.0:
        return 1.0-algInstance.objectiveFunc([x])
    else:
        return algInstance.objectiveFunc([x])

def gradient(x, algInstance):
    return algInstance.objectiveFuncSign()*algInstance.gradient([x])[0]

def centredQuadratic(x, a, c):
    return a*x*x + c

def drawMajorizer(plt, mlist, u, rhou, rhopu, colour):
    a = 0.5*rhopu/u
    c = rhou - a*u*u
    rmfv = np.vectorize(centredQuadratic, excluded={"a", "c"})
    plt.plot(mlist, rmfv(mlist, a=a, c=c), color = colour, lw = 1.0, label = "Majorizer u=" + str(u))
    
def plotResult(algInstance, uValues, label, fileName):
    xMax = 3.0
    mlist = np.linspace(-xMax, xMax, num=300)

    plt.figure(num=1, dpi=240)
    ax = plt.gca()

    colours = ["blue", "purple"]
    for u,col in zip(uValues,colours):
        drawMajorizer(plt, mlist, u, objectiveFunc(u,algInstance), gradient(u,algInstance), col)

    rmfv = np.vectorize(objectiveFunc, excluded={"algInstance"})
    plt.plot(mlist, rmfv(mlist, algInstance=algInstance), color = 'green', lw=1.0, label=label)

    ax.set_xlabel(r'r')

    plt.legend()
    plt.savefig("../../Output/" + fileName + ".png", bbox_inches='tight')
    plt.show()

paramInstance = GNC_IRLSpParams(0.0, 1.0, 0.1) # p, rscale, epsilonBase
algInstance = GNC_IRLSpMean(paramInstance, [0.0], [1.0])
plotResult(algInstance, [1.0, 2.0], 'GNC IRLSp0 influence function', "gncIrlsp0Majorizers")

paramInstance = GNC_IRLSpParams(0.5, 1.0, 0.1) # p, rscale, epsilonBase
algInstance = GNC_IRLSpMean(paramInstance, [0.0], [1.0])
plotResult(algInstance, [1.0, 2.0], 'GNC IRLSp0.5 influence function', "gncIrlsphMajorizers")

paramInstance = GNC_IRLSpParams(1.0, 1.0, 0.1) # p, rscale, epsilonBase
algInstance = GNC_IRLSpMean(paramInstance, [0.0], [1.0])
plotResult(algInstance, [1.0, 2.0], 'GNC IRLSp1 influence function', "gncIrlsp1Majorizers")

paramInstance = GNC_TLSParams(1.0, 1.0, 0.1) # p, rscale, muBase
algInstance = GNC_TLSMean(paramInstance, [0.0], [1.0])
plotResult(algInstance, [1.5, 2.0], 'GNC TLS influence function', "gncIrlspMajorizers")

paramInstance = GNC_WelschParams(1.0) # sigma
algInstance = WelschMean(paramInstance, [0.0], [1.0])
plotResult(algInstance, [1.5, 2.0], 'Welsch influence function', "welschMajorizers")

paramInstance = WelschParams(1.0) # sigma
algInstance = PseudoHuberMean(paramInstance, [0.0], [1.0])
plotResult(algInstance, [1.5, 2.0], 'Pseudo-Huber influence function', "pseudoHuberMajorizers")
    
paramInstance = GNC_WelschParams(1.0) # sigma
algInstance = GemanMcClureMean(paramInstance, [0.0], [1.0])
plotResult(algInstance, [1.0, 2.0], 'Geman-McClure influence function', "gemanMcClureMajorizers")
