import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import json
import os

from gnc_smoothie_philfm.plt_alg_vis import gncs_draw_curve

from robust_solver_apply import apply_to_data

def main(testrun:bool, output_folder:str="../../Output"):
    sigmaPop = 1.0
    p = 0.5 # 33333333

    # data range
    xgtrange = 50.0

    # number of samples used for statistics
    nSamplesBase = 200 if testrun else 50000
    minNSamples = 4 if testrun else 1000

    np.random.seed(0) # We want the numbers to be the same on each run

    studentTDOFList = [1,2,3,4,5]
    sampleSizeArray = [10,20] if testrun else [10,20,50,100]
    for N in sampleSizeArray:
        effgncwelschlist = []
        effmeanlist = []
        effhuberlist = []
        efftrimmedlist = []
        effmedianlist = []
        effgncirlsplist = []
        effrmelist = []
        data_dict = {}
        for studentTDOF in studentTDOFList:
            outputFile = '' #'../../Output/test.png' if studentTDOF == 2 else ''
            dataArray,mgt,sdgncwelsch,sdmean,sdhuber,sdtrimmed,sdmedian,sdgncirlsp,sdrme,nSamples = apply_to_data(sigmaPop, p, xgtrange, N, nSamplesBase, minNSamples, 0.0, studentTDOF=studentTDOF, outputFile=outputFile, testrun=testrun)

            outlierDict = {}
            outlierDict['data'] = dataArray
            outlierDict['popmean'] = mgt
            efficiencyDict = {}

            eff = sigmaPop*sigmaPop/(N*sdgncwelsch*sdgncwelsch)
            if not testrun:
                print("SUP-GN GNC-Welsch estimator efficiency: ", eff)

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
                print("GNC IRLS-p estimator efficiency: ", eff)

            effgncirlsplist.append(eff)
            efficiencyDict['GNC IRLS-p'] = eff

            eff = sigmaPop*sigmaPop/(N*sdrme*sdrme)
            if not testrun:
                print("Robust Mean Estimator efficiency: ", eff)

            effrmelist.append(eff)
            efficiencyDict['RME'] = eff

            outlierDict['efficiency'] = efficiencyDict
            data_dict['DOF-'+str(studentTDOF)] = outlierDict

        data_dict['nSamples'] = nSamples
        jstr = json.dumps(data_dict)
        js = json.loads(jstr)
        with open(os.path.join(output_folder, "compareStudentTN" + str(N) + ".json"), 'w', encoding='utf-8') as f:
            json.dump(js, f, ensure_ascii=False, indent=4)

        plt.close("all")
        plt.figure(num=1, dpi=240)
        plt.clf()
        ax = plt.gca()
        gncs_draw_curve(plt, effgncwelschlist,    ("IRLS",   "Welsch",      "GNC_Welsch"), xvalues=studentTDOFList)
        gncs_draw_curve(plt, effmeanlist,         ("Mean",   "Basic",       ""          ), xvalues=studentTDOFList)
        gncs_draw_curve(plt, effhuberlist,        ("IRLS",   "PseudoHuber", "Welsch"    ), xvalues=studentTDOFList)
        gncs_draw_curve(plt, efftrimmedlist,      ("Mean",   "Trimmed",     ""          ), xvalues=studentTDOFList)
        gncs_draw_curve(plt, effmedianlist,       ("Median", "Basic",       ""          ), xvalues=studentTDOFList)
        gncs_draw_curve(plt, effgncirlsplist,     ("IRLS",   "GNC_IRLSp",   "GNC_IRLSp0"), xvalues=studentTDOFList)
        gncs_draw_curve(plt, effrmelist,          ("RME",    "",            ""          ), xvalues=studentTDOFList)

        ax.set_xlabel(r'Degrees of freedom' )
        ax.set_ylabel('Relative efficiency')
        #plt.box(False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(studentTDOFList[0],studentTDOFList[len(studentTDOFList)-1])
        ax.set_ylim(0.0,1.1)

        plt.legend()
        plt.savefig(os.path.join(output_folder, "compareStudentTN" + str(N) + ".png"), bbox_inches='tight')
        if not testrun:
            plt.show()

    if testrun:
        print("student_t_solver OK")
