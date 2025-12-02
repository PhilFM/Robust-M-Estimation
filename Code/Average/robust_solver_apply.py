import numpy as np
import matplotlib.pyplot as plt
import math

from robust_mean import M_estimator

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.irls import IRLS
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.null_params import NullParams
from gnc_smoothie_philfm.gnc_irls_p_params import GNC_IRLSpParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie_philfm.pseudo_huber_influence_func import PseudoHuberInfluenceFunc
from gnc_smoothie_philfm.gnc_irls_p_influence_func import GNC_IRLSpInfluenceFunc
from gnc_smoothie_philfm.draw_functions import gncs_draw_data_points
from gnc_smoothie_philfm.plt_alg_vis import gncs_draw_vline, gncs_draw_curve

from trimmed_mean import trimmed_mean
from gncs_robust_mean import RobustMean
from weighted_mean import weighted_mean

showOthers = True

def objective_func(m, optimiser_instance):
    return optimiser_instance.objective_func([m])

def apply_to_data(sigmaPop,p,xgtrange,N,nSamplesBase,minNSamples,outlierFraction, studentTDOF = 0, outputFile = '', testrun:bool=False):
    vgncwelsch = vmean = vhuber = vtrimmed = vmedian = vgncirlsp = vrme = 0.0

    N0 = int((1.0-outlierFraction)*N+0.5)
    nSamples = max(nSamplesBase//N, minNSamples)
    if not testrun:
        print("outlierFraction=",outlierFraction," N=",N," N0=",N0," nSamples=",nSamples," studentTDOF=",studentTDOF)

    xgtborder = 3.0*sigmaPop
    dataArray = []
    for i in range(nSamples):
        mgt = np.random.rand()*xgtrange + xgtborder
        data = np.zeros((N,1))
        weight = np.zeros(N)
        goodData = []
        for j in range(N0):
            if studentTDOF > 1:
                d = mgt + np.random.standard_t(studentTDOF)
            else:
                d = np.random.normal(loc=mgt, scale=sigmaPop)

            weight[j] = 1.0
            data[j] = [d]
            goodData.append([weight[j], [d]])

        outlierData = []
        for j in range(N-N0):
            d = np.random.rand()*(xgtrange + 2.0*xgtborder)
            weight[N0+j] = 1.0
            data[N0+j] = [d]
            outlierData.append([weight[N0+j], [d]])

        datap = {}
        datap["good"] = goodData
        datap["outlier"] = outlierData

        dataArray.append(datap)

        model_instance = RobustMean()

        gncwelschSigma = sigmaPop/p
        welschIRLSInstance = IRLS(GNC_WelschParams(WelschInfluenceFunc(), gncwelschSigma, max(xgtrange,10.0*sigmaPop), 100),
                                  model_instance, data, weight=weight, max_niterations=200)

        # for the paper let's use IRLS
        mgncwelsch = welschIRLSInstance.run()
        vgncwelsch += math.pow(mgncwelsch-mgt, 2.0)
        welschOptInstance = SupGaussNewton(GNC_WelschParams(WelschInfluenceFunc(), gncwelschSigma, max(xgtrange,10.0*sigmaPop), 100),
                                           model_instance, data, weight=weight, max_niterations=200)

        mean = weighted_mean(data, weight)
        vmean += math.pow(mean-mgt, 2.0)

        pseudoHuberSigma = sigmaPop/p
        pseudoHuberIRLSInstance = IRLS(NullParams(PseudoHuberInfluenceFunc(sigma=pseudoHuberSigma)),
                                       model_instance, data, weight=weight)
        mhuber = pseudoHuberIRLSInstance.run()
        vhuber += math.pow(mhuber-mgt, 2.0)
        pseudoHuberOptInstance = SupGaussNewton(NullParams(PseudoHuberInfluenceFunc(sigma=pseudoHuberSigma)),
                                                model_instance, data, weight=weight)

        #trimSize = N//10
        #mtrimmed = trimmed_mean(data, trimSize=trimSize)
        #vtrimmed1 += math.pow(mtrimmed-mgt, 2.0)

        trimSize = N//4
        mtrimmed = trimmed_mean(data, weight, trimSize=trimSize)
        vtrimmed += math.pow(mtrimmed-mgt, 2.0)

        median = np.median(data)
        vmedian += math.pow(median-mgt, 2.0)

        gncIrlsp_rscale = 1.0/xgtrange
        gncIrlsp_epsilon_base = gncIrlsp_rscale*sigmaPop
        gncIrlsp_epsilon_limit = 1.0
        gncIrlsp_p = 0.0
        gncIrlsp_beta = 0.8
        gncIrlspIRLSInstance = IRLS(GNC_IRLSpParams(GNC_IRLSpInfluenceFunc(),
                                                    gncIrlsp_p, gncIrlsp_rscale, gncIrlsp_epsilon_base, gncIrlsp_epsilon_limit, gncIrlsp_beta),
                                    model_instance, data, weight=weight)
        mgncirlsp = gncIrlspIRLSInstance.run()
        vgncirlsp += math.pow(mgncirlsp-mgt, 2.0)
        gncIrlspOptInstance = SupGaussNewton(GNC_IRLSpParams(GNC_IRLSpInfluenceFunc(),
                                                             gncIrlsp_p, gncIrlsp_rscale, gncIrlsp_epsilon_base, gncIrlsp_epsilon_limit, gncIrlsp_beta),
                                             model_instance, data, weight=weight)

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

                # allow border
                drange = dmax-dmin
                xMin = dmin - 0.05*drange
                xMax = dmax + 0.05*drange

            mlist = np.linspace(xMin, xMax, num=300)

            for mx in mlist:
                yMax = max(yMax, objective_func(mx, welschOptInstance))

            yMin *= 1.1 # allow for a small border
            yMax *= 1.1 # allow for a small border            

            plt.figure(num=1, dpi=240)
            ax = plt.gca()
            #plt.box(False)
            ax.set_ylim((yMin, yMax))

            rmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
            pltAlgVis.drawCurve(plt, rmfv(mlist, optimiser_instance=welschOptInstance), ("IRLS", "Welsch", "GNC_Welsch"), xvalues=mlist, drawMarkers=False, hlightXValue=mgncwelsch, ax=ax)

            hmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
            hmfvScaled = hmfv(mlist, optimiser_instance=pseudoHuberOptInstance)
            hmfvScaled *= 0.5
            pltAlgVis.drawCurve(plt, hmfvScaled, ("IRLS", "PseudoHuber", "Welsch"), xvalues=mlist, drawMarkers=False, hlightXValue=mhuber, ax=ax)

            gmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
            gmfvScaled = gmfv(mlist, optimiser_instance=gncIrlspOptInstance)
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

