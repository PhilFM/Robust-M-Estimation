import numpy as np
import matplotlib.pyplot as plt
import math
import sys

sys.path.append("Welsch")
from WelschMean import WelschMean

sys.path.append("PseudoHuber")
from PseudoHuberMean import PseudoHuberMean

sys.path.append("GNC_IRLSp")
from GNC_IRLSpMean import GNC_IRLSpMean

sys.path.append("Trimmed")
from trimmedMean import trimmedMean

sys.path.append("../Library")
from IRLS import IRLS
from weightedMean import weightedMean
from GNC_WelschParams import GNC_WelschParams
from WelschParams import WelschParams
from GNC_IRLSpParams import GNC_IRLSpParams
from drawFunctions import drawDataPoints
import pltAlgVis

from robust_mean import M_estimator
from drawFunctions import drawDataPoints

showOthers = True

def objectiveFunc(m, algInstance):
    return algInstance.objectiveFunc([m])

def applyToData(sigmaPop,p,xgtrange,N,nSamplesBase,minNSamples,outlierFraction, studentTDOF = 0, outputFile = ''):
    vgncwelsch = vmean = vhuber = vtrimmed = vmedian = vgncirlsp = vrme = 0.0

    N0 = int((1.0-outlierFraction)*N+0.5)
    nSamples = max(nSamplesBase//N, minNSamples)
    print("outlierFraction=",outlierFraction," N=",N," N0=",N0," nSamples=",nSamples," studentTDOF=",studentTDOF)
    xgtborder = 3.0*sigmaPop
    dataArray = []
    for i in range(nSamples):
        mgt = np.random.rand()*xgtrange + xgtborder
        data = np.zeros(N)
        weight = np.zeros(N)
        goodData = []
        for j in range(N0):
            if studentTDOF > 1:
                d = mgt + np.random.standard_t(studentTDOF)
            else:
                d = np.random.normal(loc=mgt, scale=sigmaPop)

            weight[j] = 1.0
            data[j] = d
            goodData.append([weight[j], data[j]])

        outlierData = []
        for j in range(N-N0):
            d = np.random.rand()*(xgtrange + 2.0*xgtborder)
            weight[N0+j] = 1.0
            data[N0+j] = d
            outlierData.append([weight[N0+j], data[N0+j]])

        datap = {}
        datap["good"] = goodData
        datap["outlier"] = outlierData

        dataArray.append(datap)
        gncwelschSigma = sigmaPop/p
        welschParamInstance = GNC_WelschParams(gncwelschSigma, max(xgtrange,10.0*sigmaPop), 100)
        welschMeanInstance = WelschMean(welschParamInstance, data, weight)

        # for the paper let's use IRLS
        mgncwelsch = IRLS(welschMeanInstance, maxNIterations=200).run()
        #print("mgncwelsch=",mgncwelsch)

        #print("data: ", data, " sigmaPop: ", sigmaPop, " sigma=", gncwelschSigma, " mgncwelsch=", mgncwelsch, " mgt=", mgt)
        vgncwelsch += math.pow(mgncwelsch-mgt, 2.0)

        #print("Result: m=", mgncwelsch)
        mean = weightedMean(data, weight)
        vmean += math.pow(mean-mgt, 2.0)

        pseudoHuberSigma = sigmaPop/p
        pseudoHuberParamInstance = WelschParams(pseudoHuberSigma)
        pseudoHuberMeanInstance = PseudoHuberMean(pseudoHuberParamInstance, data, weight)
        mhuber = IRLS(pseudoHuberMeanInstance).run()
        #print("mhuber=",mhuber)
        vhuber += math.pow(mhuber-mgt, 2.0)

        #trimSize = N//10
        #print("trimSize=",trimSize)
        #mtrimmed = trimmedMean(data, trimSize=trimSize)
        #vtrimmed1 += math.pow(mtrimmed-mgt, 2.0)

        trimSize = N//4
        #print("trimSize=",trimSize)
        mtrimmed = trimmedMean(data, weight, trimSize=trimSize)
        vtrimmed += math.pow(mtrimmed-mgt, 2.0)

        median = np.median(data)
        vmedian += math.pow(median-mgt, 2.0)

        gncIrlsp_rscale = 1.0/xgtrange
        gncIrlsp_epsilonBase = gncIrlsp_rscale*sigmaPop
        gncIrlsp_epsilonLimit = 1.0
        gncIrlsp_p = 0.0
        gncIrlsp_beta = 0.8
        gncIrlspParamInstance = GNC_IRLSpParams(gncIrlsp_p, gncIrlsp_rscale, gncIrlsp_epsilonBase, gncIrlsp_epsilonLimit, gncIrlsp_beta)
        gncIrlspMeanInstance = GNC_IRLSpMean(gncIrlspParamInstance, data, weight)
        mgncirlsp = IRLS(gncIrlspMeanInstance).run()
        vgncirlsp += math.pow(mgncirlsp-mgt, 2.0)

        #print("unweightedData:",unweightedData)
        mrme = M_estimator(data, beta=1)
        vrme += math.pow(mrme-mgt, 2.0)

        if len(outputFile) > 0:
            # get min and max of data
            yMin = yMax = 0.0
            xMin = xMax = None
            # override x limit 
            #xMin = 0.7
            #xMax = 1.3

            if xMin == None:
                dmin = dmax = data[0]
                for d in data:
                    dmin = min(dmin, d)
                    dmax = max(dmax, d)
                    #print("d=", d[1], " min/max=", dmin, dmax)

                # allow border
                drange = dmax-dmin
                xMin = dmin - 0.05*drange
                xMax = dmax + 0.05*drange

            #print("xMin=", xMin, " xMax=", xMax)
            mlist = np.linspace(xMin, xMax, num=300)

            for mx in mlist:
                yMax = max(yMax, objectiveFunc(mx, welschMeanInstance))

            #print("yMin=", yMin, " yMax=", yMax)
            yMin *= 1.1 # allow for a small border
            yMax *= 1.1 # allow for a small border            

            plt.figure(num=1, dpi=240)
            ax = plt.gca()
            #plt.box(False)
            ax.set_ylim((yMin, yMax))

            rmfv = np.vectorize(objectiveFunc, excluded={"algInstance"})
            pltAlgVis.drawCurve(plt, rmfv(mlist, algInstance=welschMeanInstance), ("IRLS", "Welsch", "GNC_Welsch"), xvalues=mlist, drawMarkers=False, hlightXValue=mgncwelsch, ax=ax)

            hmfv = np.vectorize(objectiveFunc, excluded={"algInstance"})
            hmfvScaled = hmfv(mlist, algInstance=pseudoHuberMeanInstance)
            hmfvScaled *= 0.5
            pltAlgVis.drawCurve(plt, hmfvScaled, ("IRLS", "PseudoHuber", "Welsch"), xvalues=mlist, drawMarkers=False, hlightXValue=mhuber, ax=ax)

            gmfv = np.vectorize(objectiveFunc, excluded={"algInstance"})
            gmfvScaled = gmfv(mlist, algInstance=gncIrlspMeanInstance)
            gmfvScaled *= 0.1
            pltAlgVis.drawCurve(plt, gmfvScaled, ("IRLS", "GNC_IRLSp", "GNC_IRLSp0"), xvalues=mlist, drawMarkers=False, hlightXValue=mgncirlsp, ax=ax)

            pltAlgVis.drawVLine(plt, mgt, ("GroundTruth", "", ""))
            pltAlgVis.drawVLine(plt, mgncwelsch, ("IRLS", "Welsch", "GNC_Welsch"), useLabel=False)

            if showOthers:
                pltAlgVis.drawVLine(plt, mean,      ("Mean",   "Basic",       ""          ))
                pltAlgVis.drawVLine(plt, mhuber,    ("IRLS",   "PseudoHuber", "Welsch"    ), useLabel=False)
                pltAlgVis.drawVLine(plt, mtrimmed,  ("Mean",   "Trimmed",     ""          ))
                pltAlgVis.drawVLine(plt, median,    ("Median", "Basic",       ""          ))
                pltAlgVis.drawVLine(plt, mgncirlsp, ("IRLS",   "GNC_IRLSp",   "GNC_IRLSp0"), useLabel=False)
                pltAlgVis.drawVLine(plt, mrme,      ("RME",    "",            ""          ))

            drawDataPoints(plt, data, weight, xMin, xMax, N0)

            plt.legend()
            plt.savefig(outputFile, bbox_inches='tight')
            plt.show()            

    return dataArray,mgt,math.sqrt(vgncwelsch/nSamples),math.sqrt(vmean/nSamples),math.sqrt(vhuber/nSamples),math.sqrt(vtrimmed/nSamples),math.sqrt(vmedian/nSamples),math.sqrt(vgncirlsp/nSamples),math.sqrt(vrme/nSamples),nSamples

