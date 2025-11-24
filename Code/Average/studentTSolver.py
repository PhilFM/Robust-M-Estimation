import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import json
import sys

from robustSolverApply import applyToData

sys.path.append("../Library")
import pltAlgVis

# data is a list of [weight, value] pairs

sigmaPop = 1.0
p = 0.5 # 33333333

# data range
xgtrange = 50.0

# number of samples used for statistics
nSamplesBase = 50000
minNSamples = 1000

np.random.seed(0) # We want the numbers to be the same on each run

studentTDOFList = [1,2,3,4,5]
for N in [10,20,50,100]:
    effgncwelschlist = []
    effmeanlist = []
    effhuberlist = []
    efftrimmedlist = []
    effmedianlist = []
    effgncirlsplist = []
    effrmelist = []
    dataDict = {}
    for studentTDOF in studentTDOFList:
        outputFile = '' #'test.png' if studentTDOF == 2 else ''
        dataArray,mgt,sdgncwelsch,sdmean,sdhuber,sdtrimmed,sdmedian,sdgncirlsp,sdrme,nSamples = applyToData(sigmaPop, p, xgtrange, N, nSamplesBase, minNSamples, 0.0, studentTDOF=studentTDOF, outputFile=outputFile)

        outlierDict = {}
        outlierDict['data'] = dataArray
        outlierDict['popmean'] = mgt
        efficiencyDict = {}

        eff = sigmaPop*sigmaPop/(N*sdgncwelsch*sdgncwelsch)
        print("SUP-GN GNC-Welsch estimator efficiency: ", eff)
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
        print("GNC IRLS-p estimator efficiency: ", eff)
        effgncirlsplist.append(eff)
        efficiencyDict['GNC IRLS-p'] = eff

        eff = sigmaPop*sigmaPop/(N*sdrme*sdrme)
        print("Robust Mean Estimator efficiency: ", eff)
        effrmelist.append(eff)
        efficiencyDict['RME'] = eff

        outlierDict['efficiency'] = efficiencyDict
        dataDict['DOF-'+str(studentTDOF)] = outlierDict

    dataDict['nSamples'] = nSamples
    jstr = json.dumps(dataDict)
    js = json.loads(jstr)
    with open('../../Doc/RobustAverage/pictures/compareStudentTN' + str(N) + '.json', 'w', encoding='utf-8') as f:
        json.dump(js, f, ensure_ascii=False, indent=4)

    plt.figure(num=1, dpi=240)
    plt.clf()
    ax = plt.gca()
    pltAlgVis.drawCurve(plt, effgncwelschlist,    ("IRLS",   "Welsch",      "GNC_Welsch"), xvalues=studentTDOFList)
    pltAlgVis.drawCurve(plt, effmeanlist,         ("Mean",   "Basic",       ""          ), xvalues=studentTDOFList)
    pltAlgVis.drawCurve(plt, effhuberlist,        ("IRLS",   "PseudoHuber", "Welsch"    ), xvalues=studentTDOFList)
    pltAlgVis.drawCurve(plt, efftrimmedlist,      ("Mean",   "Trimmed",     ""          ), xvalues=studentTDOFList)
    pltAlgVis.drawCurve(plt, effmedianlist,       ("Median", "Basic",       ""          ), xvalues=studentTDOFList)
    pltAlgVis.drawCurve(plt, effgncirlsplist,     ("IRLS",   "GNC_IRLSp",   "GNC_IRLSp0"), xvalues=studentTDOFList)
    pltAlgVis.drawCurve(plt, effrmelist,          ("RME",    "",            ""          ), xvalues=studentTDOFList)

    ax.set_xlabel(r'Degrees of freedom' )
    ax.set_ylabel('Relative efficiency')
    #plt.box(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(studentTDOFList[0],studentTDOFList[len(studentTDOFList)-1])
    ax.set_ylim(0.0,1.1)

    plt.legend()
    plt.savefig('../../Doc/RobustAverage/pictures/compareStudentTN' + str(N) + '.png', bbox_inches='tight')
    #plt.show()
