import numpy as np
import random
import math
import matplotlib.pyplot as plt

import sys
sys.path.append("../../Library")
from IRLS import IRLS
from GNC_WelschParams import GNC_WelschParams

from WelschMean import WelschMean

sigmaBase = 1.0
sigmaLimit = 500.0
numSigmaSteps = 100
smallVal = 1.e-10

def smallMean(algInstance):
    return -algInstance.gradient([0.0])[0]/algInstance.secondDeriv([0.0])[0][0]

def efficiencyEstFunc(p):
    return math.pow(1.0 + 2.0*p*p, 1.5)*math.pow(1.0 + p*p, -3.0)

def efficiencyEstFuncN(p,N):
    numerator = N*math.pow(1.0 + 2.0*p*p, -1.5)
    denom1 = (1.0 + 3.0*p*p*p*p + 2.0*p*p)*math.pow(1.0 + 2.0*p*p, -2.5)
    denom2 = (N-1.0)*math.pow(1.0 + p*p, -3.0)
    return (denom1 + denom2)/numerator

for testIdx in range(0):
    N = 100
    data = np.zeros(N)
    weight = np.zeros(N)
    sigmaPop = 0.4
    for i in range(N):
        data[i] = random.gauss(0.0, sigmaPop)
        weight[i] = 1.0

    paramInstance = GNC_WelschParams(sigmaBase, sigmaLimit, numSigmaSteps)
    algInstance = WelschMean(paramInstance, data, weight)
    m1 = IRLS(algInstance, maxNIterations=200).run()
    m2 = smallMean(algInstance)

    print("m1=",m1," m2=",m2)

plt.figure(num=1, dpi=240)
ax = plt.gca()
ax.set_xlabel(r'$p$')
ax.set_ylabel('Efficiency')

pmax = 1.0
splist = np.linspace(0, pmax, num=30)

nSamples = 3000

Narray = [5,10,50,100]
colArray = ['magenta','r','g','cyan']
for N,col in zip(Narray,colArray):
    effData = []
    for sigmaPop in splist:
        mstot = 0.0
        lsstot = 0.0
        semx2s2 = 0.0
        sx2emx2s2 = 0.0
        sx4emx2s2 = 0.0
        semx22s2 = 0.0
        sx2emx22s2 = 0.0
        invVariance = 1.0/(sigmaBase*sigmaBase)
        smallMeanVar = 0.0
        smallMeanVarEst = 0.0
        smallMeanVarNumEst = 0.0
        smallMeanVarDenEst = 0.0
        for testIdx in range(nSamples):
            data = np.zeros(N)
            weight = np.zeros(N)
            for i in range(N):
                data[i] = random.gauss(0.0, sigmaPop)
                weight[i] = 1.0
            
            paramInstance = GNC_WelschParams(sigmaBase)
            algInstance = WelschMean(paramInstance, data, weight)
            m = smallMean(algInstance)
            mstot += m*m

            lsm = algInstance.weightedFit(weight)
            lsstot += lsm*lsm

            sxemx22s2 = 0.0
            sfid = 0.0
            for i in range(N):
                x = data[i]
                semx2s2 += math.exp(-x*x*invVariance)
                sx2emx2s2 += x*x*math.exp(-x*x*invVariance)
                sx4emx2s2 += x*x*x*x*math.exp(-x*x*invVariance)
                semx22s2 += math.exp(-0.5*x*x*invVariance)
                sx2emx22s2 += x*x*math.exp(-0.5*x*x*invVariance)

                sxemx22s2 += x*math.exp(-0.5*x*x*invVariance)
                sfid += (1.0 - x*x*invVariance)*math.exp(-0.5*x*x*invVariance)
            
            m2 = smallMean(algInstance)
            #print("m2=",m2," m2p=",sxemx22s2/sfid)
            smallMeanVar += m2*m2
            smallMeanVarEst += math.pow(sxemx22s2/sfid, 2.0)
            smallMeanVarNumEst += sxemx22s2*sxemx22s2
            smallMeanVarDenEst += sfid*sfid

        semx2s2 /= nSamples*N
        sx2emx2s2 /= nSamples*N
        sx4emx2s2 /= nSamples*N
        semx22s2 /= nSamples*N
        sx2emx22s2 /= nSamples*N
        smallMeanVar /= nSamples
        smallMeanVarEst /= nSamples
        smallMeanVarNumEst /= nSamples
        smallMeanVarDenEst /= nSamples

        p = sigmaPop/sigmaBase
        var = N*mstot/nSamples
        lsvar = N*lsstot/nSamples
        #print("sigmaPop=",sigmaPop," var=",var, " est=",sigmaBase*sigmaBase*p*p*math.sqrt(1.0+p*p)," lsvar",lsvar," est=",sigmaBase*sigmaBase*p*p)
        effData.append((lsvar+smallVal)/(var+smallVal))

        pA = p/math.sqrt(1.0+2.0*p*p)
        pB = p/math.sqrt(1.0+p*p)
        root2pi = math.sqrt(2.0*math.pi)
        #tNum = N*sigmaBase*sigmaBase*sigmaBase*pa*pa*pa*root2pi
        #tDen = N*(root2pi*pA*sigmaBase + 3.0*root2pi*math.pow(pA,5.0)*sigmaBase - 2.0*root2pi*math.pow(pA,3.0)*sigmaBase) + 0.5*N*(N-1)*(root2pi*root2pi*pB*pB)

        semx2s2Est = math.pow(1.0+2.*p*p, -0.5)
        sx2emx2s2Est = p*p*math.pow(1.0+2.*p*p, -1.5)*sigmaBase*sigmaBase
        sx4emx2s2Est = 3.0*p*p*p*p*math.pow(1.0+2.*p*p, -2.5)*sigmaBase*sigmaBase*sigmaBase*sigmaBase
        semx22s2Est = math.pow(1.0+p*p, -0.5)
        sx2emx22s2Est = p*p*math.pow(1.0+p*p, -1.5)*sigmaBase*sigmaBase
        #print("E(e^(-x^2/s^2)) = ", semx2s2Est, " est = ", semx2s2, " ratio = ", semx2s2Est/semx2s2)
        #print("E(x^2*e^(-x^2/s^2)) = ", sx2emx2s2Est, " est = ", sx2emx2s2, " ratio = ", sx2emx2s2Est/sx2emx2s2)
        #print("E(x^4*e^(-x^2/s^2)) = ", sx4emx2s2Est, " est = ", sx4emx2s2, " ratio = ", sx4emx2s2Est/sx4emx2s2)
        #print("E(e^(-x^2/(2*s^2))) = ", semx22s2Est, " est = ", semx22s2, " ratio = ", semx22s2Est/semx22s2)
        #print("E(x^2*e^(-x^2/(2*s^2))) = ", sx2emx22s2Est, " est = ", sx2emx22s2, " ratio = ", sx2emx22s2Est/sx2emx22s2)

        # calculate asymptotic efficiency
        numerator = N*sx2emx2s2Est
        denom1 = N*(semx2s2Est + sx4emx2s2Est*invVariance*invVariance - 2.0*sx2emx2s2Est*invVariance)
        denom2 = N*(N-1.0)*(semx22s2Est*semx22s2Est + sx2emx22s2Est*sx2emx22s2Est*invVariance*invVariance - 2.0*sx2emx22s2Est*semx22s2Est*invVariance)
        #print("num=",numerator," den1=",denom1," den2=",denom2)
        numeratorp = p*p*math.pow(1.0 + 2.0*p*p, -1.5)
        denom1p = (1.0 + 3.0*p*p*p*p + 2.0*p*p)*math.pow(1.0 + 2.0*p*p, -2.5)
        denom2p = (N-1.0)*math.pow(1.0 + p*p, -3.0)
        #print("nump=",N*numeratorp," den1p=",N*denom1p," den2p=",N*denom2p)

        smallMeanVarEst2 = numerator/(denom1+denom2)
        smallMeanVarEst3 = numeratorp/(denom1p+denom2p)
        smallMeanVarEst4 = smallMeanVarNumEst/smallMeanVarDenEst
        #print("Small mean variance = ", N*smallMeanVar, " est = ", N*smallMeanVarEst, " est2 = ", N*smallMeanVarEst2, " est3 = ", N*smallMeanVarEst3, " est4 = ", N*smallMeanVarEst4)
        #print("Ratio: ", smallMeanVarEst2/smallMeanVarEst, " numEst=", smallMeanVarNumEst," denEst=", smallMeanVarDenEst)
        print("p=",p, " asymptotic efficiency=", (smallVal + sigmaPop*sigmaPop/N)/(smallVal + smallMeanVarEst2), " est=", efficiencyEstFunc(p), " estN=", efficiencyEstFuncN(p,N))

    plt.plot(splist, effData, color = col, lw = 1.0, label = '$N=$' + str(N), marker = 'o', markersize = 2.0)
    #hmfv = np.vectorize(efficiencyEstFuncN, excluded={"N"})
    #mlist = np.linspace(0, pmax, num=300)
    #plt.plot(mlist, hmfv(mlist, N=N), color = col, lw = 1.0, linestyle = 'dashed', )

hmfv = np.vectorize(efficiencyEstFunc)
mlist = np.linspace(0, pmax, num=300)
plt.plot(mlist, hmfv(mlist), color = 'b', lw = 1.0, linestyle = 'dashed', label = 'asymptotic')

plt.legend()
plt.savefig('../../../Output/welschEfficiency.png', bbox_inches='tight')
plt.show()
