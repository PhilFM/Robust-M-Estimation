import numpy as np
import math
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
from IRLS import IRLS
from GNC_IRLSpParams import GNC_IRLSpParams
from GNC_TLSParams import GNC_TLSParams

def objectiveFunc(m, meanInstance):
    return meanInstance.objectiveFunc([m])

np.random.seed(0) # We want the numbers to be the same on each run

N = 10
xgtrange = 10.0
sigmaPop = 1.0
xgtborder = 3.0*sigmaPop
outlierFraction = 0.0
N0 = int((1.0-outlierFraction)*N+0.5)
est_p = 0.6666667
sigma = sigmaPop/est_p
plot_results = False

def derivChecker(meanAlgInstance):
    smallDiff = 1.e-5
    gradVal = meanAlgInstance.gradient([x])[0]
    gradNum = 0.5*(meanAlgInstance.objectiveFunc([x+smallDiff]) - meanAlgInstance.objectiveFunc([x-smallDiff]))/smallDiff
    print("   Gradient check      ",gradVal,gradNum,gradVal-gradNum)
    if abs(gradVal-gradNum) > 1.e-7:
        return False

    secondDerivVal = meanAlgInstance.secondDeriv([x])[0][0]
    secondDerivNum = 0.5*(meanAlgInstance.gradient([x+smallDiff])[0]
                          - meanAlgInstance.gradient([x-smallDiff])[0])/smallDiff
    print("   2nd derivative check",secondDerivVal,secondDerivNum,secondDerivVal-secondDerivNum)
    if abs(secondDerivVal-secondDerivNum) > 1.e-8:
        return False

    weightedDerivZero = meanAlgInstance.weightedDeriv([x], 0.0)[0][0]
    weightedDerivZeroCheck = meanAlgInstance.weightSum([x])
    print("   weighted deriv zero ",weightedDerivZero,weightedDerivZeroCheck,weightedDerivZero-weightedDerivZeroCheck)
    if abs(weightedDerivZero-weightedDerivZeroCheck) > 1.e-10:
        return False

    weightedDerivOne = meanAlgInstance.weightedDeriv([x], 1.0)[0][0]
    print("   weighted deriv one  ",weightedDerivOne,secondDerivVal,weightedDerivOne-secondDerivVal)
    if abs(weightedDerivOne-secondDerivVal) > 1.e-10:
        return False

    return True

allGood = True
for testIdx in range(0,10):
    mgt = np.random.rand()*xgtrange + xgtborder
    data = np.zeros(N)
    weight = np.zeros(N)
    for j in range(N0):
        weight[j] = 1.0
        data[j] = np.random.normal(loc=mgt, scale=sigmaPop)

    for j in range(N-N0):
        weight[N0+j] = 1.0
        data[N0+j] = np.random.rand()*(xgtrange + 2.0*xgtborder)

    x = mgt + 2.0*(np.random.rand()-0.5)*sigmaPop

    gncWelschParamInstance = GNC_WelschParams(sigma)

    print("")
    print("Welsch:")
    if derivChecker(WelschMean(gncWelschParamInstance, data, weight)) is False:
        allGood = False
        break

    print("")
    print("PseudoHuber:")
    if derivChecker(PseudoHuberMean(gncWelschParamInstance, data, weight)) is False:
        allGood = False
        break

    print("")
    print("GemanMcClure:")
    if derivChecker(GemanMcClureMean(gncWelschParamInstance, data, weight)) is False:
        allGood = False
        break

    print("")
    print("GNC IRLS-p:")
    p = 0.0
    rscale = 1.0/xgtrange
    epsilon = rscale*sigmaPop
    gncIrlspParamInstance = GNC_IRLSpParams(p, rscale, epsilon)
    gncIrlspMeanInstance = GNC_IRLSpMean(gncIrlspParamInstance, data, weight)
    if derivChecker(gncIrlspMeanInstance) is False:
        allGood = False
        break

    xMin = -4.0
    xMax = xgtrange+4.0
    mlist = np.linspace(xMin, xMax, num=300)

    if plot_results:
        plt.figure(num=1, dpi=240)
        ax = plt.gca()
        rmfv = np.vectorize(gncIrlspMeanFunc, excluded={"p","epsilon","rscale","data","weight"})
        plt.plot(mlist, rmfv(mlist, p=p, epsilon=epsilon, rscale=rscale, data=data, weight=weight), color = 'green', lw = 1.0)
        plt.axvline(x=x, color = 'green',   label = 'Mean', lw = 1.0, linestyle = 'dotted')
        plt.legend()
        plt.show()
    
    print("")
    print("GNC TLS:")
    p = 0.0
    rscale = 1.0/xgtrange
    c = 0.1
    mu = 0.5

    gncTlsParamInstance = GNC_TLSParams(c, rscale, mu)
    gncTlsMeanInstance = GNC_TLSMean(gncTlsParamInstance, data, weight)
    if derivChecker(gncTlsMeanInstance) is False:
        allGood = False
        break

    if plot_results:
        plt.figure(num=1, dpi=240)
        ax = plt.gca()
        rmfv = np.vectorize(objectiveFunc, excluded={"meanInstance"})
        plt.plot(mlist, rmfv(mlist, meanInstance=gncTlsMeanInstance), color = 'green', lw = 1.0)
        plt.axvline(x=x, color = 'green',   label = 'Mean', lw = 1.0, linestyle = 'dotted')
        plt.legend()
        plt.show()
    
if allGood:
    print("ALL DERIVATIVES OK!!")
else:
    print("Derivative failure")
