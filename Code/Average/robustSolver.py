import numpy as np
import matplotlib.pyplot as plt
import json
import sys

from robustSolverApply import applyToData

sys.path.append("../Library")
import pltAlgVis

sigmaPop = 1.0

# data is a list of [weight, value] pairs

p = 0.66666667

# number of samples used for statistics
nSamplesBase = 50000
minNSamples = 1000

np.random.seed(0) # We want the numbers to be the same on each run

outlierFractionList = [0.0,0.1,0.2,0.3,0.4,0.5]#,0.6,0.7,0.8]
for xgtrange in [3.0,5.0,10.0,30.0,100.0]:
    for N in [10,30,100,1000]:
        effgncwelschlist = []
        effmeanlist = []
        effhuberlist = []
        efftrimmedlist = []
        effmedianlist = []
        effgncirlsplist = []
        effrmelist = []
        dataDict = {}
        for outlierFraction in outlierFractionList:
            outputFile = '' # 'solver-' + str(int(xgtrange)) + "-" + str(N) + "-" + str(int(100.0*outlierFraction)) + ".png"
            dataArray,mgt,sdgncwelsch,sdmean,sdhuber,sdtrimmed,sdmedian,sdgncirlsp,sdrme,nSamples = applyToData(sigmaPop, p, xgtrange, N, nSamplesBase, minNSamples, outlierFraction, outputFile=outputFile)

            outlierDict = {}
            outlierDict['data'] = dataArray
            outlierDict['popmean'] = mgt
            efficiencyDict = {}

            eff = sigmaPop*sigmaPop/(N*sdgncwelsch*sdgncwelsch)
            print("GNC Welsch estimator efficiency: ", eff)
            effgncwelschlist.append(eff)
            efficiencyDict['GNC Welsch'] = eff

            eff = sigmaPop*sigmaPop/(N*sdmean*sdmean)
            print("Mean efficiency: ", eff)
            effmeanlist.append(eff)
            efficiencyDict['mean'] = eff

            eff = sigmaPop*sigmaPop/(N*sdhuber*sdhuber)
            print("Pseudo-Huber estimator efficiency: ", eff)
            effhuberlist.append(eff)
            efficiencyDict['Pseudo-Huber'] = eff

            eff = sigmaPop*sigmaPop/(N*sdtrimmed*sdtrimmed)
            print("Trimmed mean 50% efficiency: ", eff)
            efftrimmedlist.append(eff)
            efficiencyDict['trimmed mean 50%'] = eff

            eff = sigmaPop*sigmaPop/(N*sdmedian*sdmedian)
            print("Median efficiency: ", eff)
            effmedianlist.append(eff)
            efficiencyDict['median'] = eff

            eff = sigmaPop*sigmaPop/(N*sdgncirlsp*sdgncirlsp)
            print("GNC IRLS-p=0 estimator efficiency: ", eff)
            effgncirlsplist.append(eff)
            efficiencyDict['GNC IRLS-p'] = eff

            eff = sigmaPop*sigmaPop/(N*sdrme*sdrme)
            print("Robust Mean Estimator efficiency: ", eff)
            effrmelist.append(eff)
            efficiencyDict['RME'] = eff

            outlierDict['efficiency'] = efficiencyDict
            dataDict['outlierFraction-'+str(outlierFraction)] = outlierDict

        dataDict['nSamples'] = nSamples
        jstr = json.dumps(dataDict)
        js = json.loads(jstr)
        with open('../../Output/compareN' + str(N) + '-range' + str(int(xgtrange)) + '.json', 'w', encoding='utf-8') as f:
            json.dump(js, f, ensure_ascii=False, indent=4)

        plt.figure(num=1, dpi=240)
        plt.clf()
        ax = plt.gca()
        pltAlgVis.drawCurve(plt, effgncwelschlist,    ("IRLS",   "Welsch",      "GNC_Welsch"), xvalues=outlierFractionList)
        pltAlgVis.drawCurve(plt, effmeanlist,         ("Mean",   "Basic",       ""          ), xvalues=outlierFractionList)
        pltAlgVis.drawCurve(plt, effhuberlist,        ("IRLS",   "PseudoHuber", "Welsch"    ), xvalues=outlierFractionList)
        pltAlgVis.drawCurve(plt, efftrimmedlist,      ("Mean",   "Trimmed",     ""          ), xvalues=outlierFractionList)
        pltAlgVis.drawCurve(plt, effmedianlist,       ("Median", "Basic",       ""          ), xvalues=outlierFractionList)
        pltAlgVis.drawCurve(plt, effgncirlsplist,     ("IRLS",   "GNC_IRLSp",   "GNC_IRLSp0"), xvalues=outlierFractionList)
        pltAlgVis.drawCurve(plt, effrmelist,          ("RME",    "",            ""          ), xvalues=outlierFractionList)

        ax.set_xlabel(r'Outlier fraction' )
        ax.set_ylabel('Relative efficiency')
        #plt.box(False)
        ax.set_xlim(0.0,outlierFractionList[len(outlierFractionList)-1])
        ax.set_ylim(0.0,1.1)

        plt.legend()
        plt.savefig('../../Output/compareN' + str(N) + '-range' + str(int(xgtrange)) + '.png', bbox_inches='tight')
        #plt.show()
