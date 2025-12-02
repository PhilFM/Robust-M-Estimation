import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import argparse

from robust_solver_apply import apply_to_data

sys.path.append("../Library")
import pltAlgVis

def main(testrun:bool):
    sigmaPop = 1.0
    p = 0.66666667

    # number of samples used for statistics
    nSamplesBase = 100 if testrun else 50000
    minNSamples = 4 if testrun else 1000

    np.random.seed(0) # We want the numbers to be the same on each run

    outlierFractionList = [0.0,0.2,0.5] if testrun else [0.0,0.1,0.2,0.3,0.4,0.5]
    plot_count = 1
    for xgtrange in [3.0,5.0,10.0,30.0,100.0]:
        sampleSizeArray = [10] if testrun else [10,30,100,1000]
        for N in sampleSizeArray:
            effgncwelschlist = []
            effmeanlist = []
            effhuberlist = []
            efftrimmedlist = []
            effmedianlist = []
            effgncirlsplist = []
            effrmelist = []
            data_dict = {}
            for outlierFraction in outlierFractionList:
                outputFile = '' # '../../Output/solver-' + str(int(xgtrange)) + "-" + str(N) + "-" + str(int(100.0*outlierFraction)) + ".png"
                dataArray,mgt,sdgncwelsch,sdmean,sdhuber,sdtrimmed,sdmedian,sdgncirlsp,sdrme,nSamples = apply_to_data(sigmaPop, p, xgtrange, N, nSamplesBase, minNSamples, outlierFraction, outputFile=outputFile, testrun=testrun)

                outlierDict = {}
                outlierDict['data'] = dataArray
                outlierDict['popmean'] = mgt
                efficiencyDict = {}

                eff = sigmaPop*sigmaPop/(N*sdgncwelsch*sdgncwelsch)
                if not testrun:
                    print("GNC Welsch estimator efficiency: ", eff)

                effgncwelschlist.append(eff)
                efficiencyDict['GNC Welsch'] = eff

                eff = sigmaPop*sigmaPop/(N*sdmean*sdmean)
                if not testrun:
                    print("Mean efficiency: ", eff)

                effmeanlist.append(eff)
                efficiencyDict['mean'] = eff

                eff = sigmaPop*sigmaPop/(N*sdhuber*sdhuber)
                if not testrun:
                    print("Pseudo-Huber estimator efficiency: ", eff)

                effhuberlist.append(eff)
                efficiencyDict['Pseudo-Huber'] = eff

                eff = sigmaPop*sigmaPop/(N*sdtrimmed*sdtrimmed)
                if not testrun:
                    print("Trimmed mean 50% efficiency: ", eff)

                efftrimmedlist.append(eff)
                efficiencyDict['trimmed mean 50%'] = eff

                eff = sigmaPop*sigmaPop/(N*sdmedian*sdmedian)
                if not testrun:
                    print("Median efficiency: ", eff)

                effmedianlist.append(eff)
                efficiencyDict['median'] = eff

                eff = sigmaPop*sigmaPop/(N*sdgncirlsp*sdgncirlsp)
                if not testrun:
                    print("GNC IRLS-p=0 estimator efficiency: ", eff)

                effgncirlsplist.append(eff)
                efficiencyDict['GNC IRLS-p'] = eff

                eff = sigmaPop*sigmaPop/(N*sdrme*sdrme)
                if not testrun:
                    print("Robust Mean Estimator efficiency: ", eff)

                effrmelist.append(eff)
                efficiencyDict['RME'] = eff

                outlierDict['efficiency'] = efficiencyDict
                data_dict['outlierFraction-'+str(outlierFraction)] = outlierDict

            data_dict['nSamples'] = nSamples
            jstr = json.dumps(data_dict)
            js = json.loads(jstr)
            with open('../../Output/compareN' + str(N) + '-range' + str(int(xgtrange)) + '.json', 'w', encoding='utf-8') as f:
                json.dump(js, f, ensure_ascii=False, indent=4)

            plt.figure(num=plot_count, dpi=240)
            plot_count += 1
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
            if not testrun:
                plt.show()

    if testrun:
        print("OK")

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--testrun', action="store_true", default=False)
args = parser.parse_args()
main(args.testrun)
