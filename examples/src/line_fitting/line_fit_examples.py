import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
import seaborn as sns
from sklearn import linear_model
from sklearn.datasets import fetch_openml
import scipy
from pathlib import Path
from robpy.regression import MMRegression
import sys

if __name__ == "__main__":
    sys.path.append("../../../pypi_package/src")
    sys.path.append("../../../pypi_package/src/gnc_smoothie/linear_model")
    sys.path.append("../../../pypi_package/src/gnc_smoothie/cython_files")

from line_fit_orthog_welsch import LineFitOrthogWelsch

from gnc_smoothie.irls import IRLS
from gnc_smoothie.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie.plt_alg_vis import gncs_draw_curve
from gnc_smoothie.linear_model.linear_regressor_welsch import LinearRegressorWelsch
from gnc_smoothie.linear_model.linear_regressor import LinearRegressor

sup_gn_q = 10.0

# check for scipy style X/y "training data/target" arguments, and convert to single data array
def convert_data(data):
    #print("Input",data)
    if isinstance(data, tuple):
        assert(len(data) == 2) # data_x, data_y
        data_x = data[0]
        data_y = data[1]
        #print("data_x=",data_x)
        #print("data_y=",data_y)
        assert(len(data_x) == len(data_y))
        data_y = np.reshape(data_y.astype(np.double), (len(data_y),1))
        data_x = np.reshape(data_x.astype(np.double), (len(data_x),1))
        return np.concatenate((data_x, data_y), axis=1)
    else:
        return data

def draw_theil_sen_line(data, x_min:float, x_max:float):
    Xnp = np.array(data[:,0]).reshape((len(data),1))
    Ynp = np.array(data[:,1])
    theil_sen = linear_model.TheilSenRegressor() #max_subpopulation=1e10)
    theil_sen.fit(X=Xnp, y=Ynp)
    coeff = theil_sen.coef_
    intercept = theil_sen.intercept_
    plt.axline((x_min, coeff[0]*x_min+intercept), (x_max, coeff[0]*x_max+intercept), color = "brown", linewidth=1.0, label="Theil-Sen fit")

def draw_mm_estimation_line(data, x_min:float, x_max:float):
    data_x = np.array(data[:,0]).reshape((len(data),1))
    data_y = np.array(data[:,1])
    #data_x = data[:,0]
    #data_y = data[:,1]
    estimator = MMRegression(prepend_intercept=False).fit(data_x, data_y)
    print("MM estimator scale=", estimator._scale)
    line_ab = estimator.model.coef_
    plt.axline((x_min, line_ab[0]*x_min+line_ab[1]), (x_max, line_ab[0]*x_max+line_ab[1]), color = "magenta", linewidth=1.0, label="MM estimation fit")

def plot_differences(diffs_welsch_sup_gn, diff_alpha_welsch_sup_gn,
                     diffs_welsch_irls, diff_alpha_welsch_irls, output_file_name:str,
                     test_idx: int, test_run:bool, output_folder:str):
    if not test_run:
        print("diffs_welsch_sup_gn:",diffs_welsch_sup_gn)
        print("diffs_welsch_irls:",diffs_welsch_irls)
        print("diff_alpha_welsch_sup_gn:",diff_alpha_welsch_sup_gn)
        print("diff_alpha_welsch_irls:",diff_alpha_welsch_irls)

    plt.close("all")
    plt.figure(num=1, dpi=240)
    plt.clf()
    ax = plt.gca()
    ax.set_xlim(0,int(max(len(diffs_welsch_sup_gn),len(diffs_welsch_irls))))

    idx = np.argmax(diff_alpha_welsch_sup_gn)
    if idx > 0:
        gncs_draw_curve(plt, diffs_welsch_sup_gn[0:idx+1], ("SupGN", "Welsch", "GNC_Welsch"),
                        lw=0.2, xvalues = np.arange(0,idx+1), add_label=False, markersize=1.0)

    gncs_draw_curve(plt, diffs_welsch_sup_gn[idx:], ("SupGN", "Welsch", "GNC_Welsch"),
                    lw=1.4, xvalues = np.arange(idx,len(diffs_welsch_sup_gn)), markersize=3.0)

    idx = np.argmax(diff_alpha_welsch_irls)
    if idx > 0:
        gncs_draw_curve(plt, diffs_welsch_irls[0:idx+1], ("IRLS",  "Welsch", "GNC_Welsch"),
                        lw=0.2, xvalues = np.arange(0,idx+1), add_label=False, markersize=1.0)

    gncs_draw_curve(plt, diffs_welsch_irls[idx:], ("IRLS",  "Welsch", "GNC_Welsch"),
                    lw=1.4, xvalues = np.arange(idx,len(diffs_welsch_irls)), markersize=3.0)

    ax.set_xlabel(r'Iteration count' )
    ax.set_ylabel(r'log(difference)')

    plt.legend()
    plt.savefig(os.path.join(output_folder, "convergence_speed_" + str(test_idx+1) + "_" + output_file_name), bbox_inches='tight')
    if not test_run:
        plt.show()

def fit_line_dependent(data:np.ndarray, x_label:str, y_label:str, sigma:float, output_file_name:str, test_run:bool, output_folder:str, *, fix_origin:bool=False):
    #print("data=",data)
    (x_min,x_max) = (min(data[:,0]),max(data[:,0]))
    (y_min,y_max) = (min(data[:,1]),max(data[:,1]))
    sigma_base = sigma/0.6667
    sigma_limit = y_max-y_min
    num_sigma_steps = 10
    max_niterations = 200
    diff_thres = 1.e-12
    line_fitter = LinearRegressorWelsch(sigma_base=sigma_base, sigma_limit=sigma_limit, num_sigma_steps=num_sigma_steps,
                                        max_niterations=max_niterations, diff_thres=diff_thres,
                                        messages_file=None if test_run else sys.stdout, debug=True)
    fit_ok = line_fitter.fit(data)
    if fit_ok:
        diffs_welsch_sup_gn = line_fitter.debug_diffs
        diff_alpha_welsch_sup_gn = np.array(line_fitter.debug_diff_alpha)

        param_instance = GNC_WelschParams(WelschInfluenceFunc(), sigma_base,
                                          sigma_limit=sigma_limit, num_sigma_steps=num_sigma_steps)
        model_instance = LinearRegressor(data[0])
        irls_instance = IRLS(param_instance, model_instance=model_instance,
                             max_niterations=max_niterations, diff_thres=diff_thres,
                             messages_file=None if test_run else sys.stdout, debug=True)
        irls_instance.fit(data) # we don't care if IRLS fails to converge
        diffs_welsch_irls = irls_instance.debug_diffs
        diff_alpha_welsch_irls = np.array(irls_instance.debug_diff_alpha)

        test_idx = 0
        plot_differences(diffs_welsch_sup_gn, diff_alpha_welsch_sup_gn,
                         diffs_welsch_irls, diff_alpha_welsch_irls, output_file_name,
                         test_idx, test_run, output_folder)

    if fit_ok:
        final_weight = line_fitter.final_weight
        (a,b) = (line_fitter.final_model[0],line_fitter.final_model[1])

        # show result
        plt.close("all")
        plt.figure(num=1, dpi=120)

        ax = plt.gca()
        if fix_origin:
            ax.set_xlim(0.0, x_max*1.02)
            ax.set_ylim(0.0, y_max*1.02)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        plt.plot(data[0][0], data[0][1], color = (1,0,0), marker='o', label="Inlier data values") # will be overwritten with corrected colour
        plt.plot(data[0][0], data[0][1], color = (0,0,1), marker='o', label="Outlier data values") # will be overwritten with corrected colour
        max_weight = max(final_weight)
        for d,w in zip(data,final_weight, strict=True):
            alpha = w/max_weight
            color = [alpha, 0.0, 1.0-alpha]
            plt.plot(d[0], d[1], color = color, marker = 'o')

        if sigma is None:
            draw_mm_estimation_line(data, x_min, x_max)
            #draw_theil_sen_line(data, x_min, x_max)
            
        if False:
            debug_line_list = line_fitter.debug_model_list
            for i,line in enumerate(debug_line_list):
                (a,b) = (-line[1][0]/line[1][1], -line[1][2]/line[1][1])
                color = (0.5+0.5*line[0], 1.0, 0.5+0.5*line[0])
                plt.axline((x_min, a*x_min+b), (x_max, a*x_max+b), color = color, linewidth=0.5, label = "Intermediate lines" if i == 0 else None)

        plt.axline((x_min, a*x_min+b), (x_max, a*x_max+b), color = "green", linewidth=1.5, label="Best fit line")

        plt.legend()
        plt.savefig(os.path.join(output_folder, "example_" + output_file_name), bbox_inches='tight')
        if not test_run:
            plt.show()
    else:
        print("Line fitter failed")

def fit_line_orthog(method:str, data:np.ndarray, x_label:str, y_label:str, sigma:float, output_file_name:str, test_run:bool, output_folder:str, *, axis_def:str=None):
    #print("data=",data)
    (x_min,x_max) = (min(data[:,0]),max(data[:,0]))
    (y_min,y_max) = (min(data[:,1]),max(data[:,1]))
    sigma_base = sigma/sup_gn_q
    sigma_limit = max(x_max-x_min, y_max-y_min)

    # orthogonal regression fitter a*x + b*y + c = 0 where a^2+b^2=1
    line_fitter = LineFitOrthogWelsch(sigma_base, sigma_limit=sigma_limit, num_sigma_steps=20,
                                      debug=False if test_run else True)
    if line_fitter.fit(data, method=method):
        final_line = line_fitter.final_line
        final_weight = line_fitter.final_weight
        (a,b) = (-final_line[0]/final_line[1],-final_line[2]/final_line[1])
        if not test_run:
            debug_line_list = line_fitter.debug_line_list

            if method == "BestAngle":
                (ref_a,ref_b) = (-line_fitter.debug_ref_line[0]/line_fitter.debug_ref_line[1],-line_fitter.debug_ref_line[2]/line_fitter.debug_ref_line[1])

            if method == "BestAngle":
                (ref_a,ref_b) = (-line_fitter.debug_ref_line[0]/line_fitter.debug_ref_line[1],-line_fitter.debug_ref_line[2]/line_fitter.debug_ref_line[1])

        # show orthogonal regression result
        plt.close("all")
        plt.figure(num=1, dpi=120)

        ax = plt.gca()
        if axis_def is not None:
            if axis_def == "fix origin":
                ax.set_xlim(0.0, x_max*1.02)
                ax.set_ylim(0.0, y_max*1.02)
            elif axis_def == "fix range":
                border_size = 0.1
                xrange = x_max-x_min
                x_min -= border_size*xrange
                x_max += border_size*xrange
                yrange = y_max-y_min
                y_min -= border_size*yrange
                y_max += border_size*yrange
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        plt.plot(data[0][0], data[0][1], color = (1,0,0), marker='o', label="Inlier data values") # will be overwritten with corrected colour
        plt.plot(data[0][0], data[0][1], color = (0,0,1), marker='o', label="Outlier data values") # will be overwritten with corrected colour
        max_weight = max(final_weight)
        for d,w in zip(data,final_weight, strict=True):
            alpha = w/max_weight
            color = [alpha, 0.0, 1.0-alpha]
            plt.plot(d[0], d[1], color = color, marker = 'o')

        if method == "BestAngle":
            draw_mm_estimation_line(data, x_min, x_max)
            #draw_theil_sen_line(data, x_min, x_max)

        if not test_run:
            for i,line in enumerate(debug_line_list):
                (a,b) = (-line[1][0]/line[1][1], -line[1][2]/line[1][1])
                color = (0.5+0.5*line[0], 1.0, 0.5+0.5*line[0])
                plt.axline((x_min, a*x_min+b), (x_max, a*x_max+b), color = color, linewidth=0.5, label = "Intermediate lines" if i == 0 else None)

            if method == "BestAngle":
                plt.axline((x_min, ref_a*x_min+ref_b), (x_max, ref_a*x_max+ref_b), color = "cyan", linewidth=1.0, label="Reference angle")

        plt.axline((x_min, a*x_min+b), (x_max, a*x_max+b), color = "green", linewidth=1.5, label="Best fit line ("+method+")")

        plt.legend()
        plt.savefig(os.path.join(output_folder, output_file_name), bbox_inches='tight')
        if not test_run:
            plt.show()

def test_anscombe(test_run:bool, output_folder:str):
    # Load the quartet
    df = sns.load_dataset("anscombe")

    # Filter for dataset III (the one with the extreme outlier)
    dataset_3 = df[df['dataset'] == 'III']
    fit_line_dependent(convert_data((dataset_3.values[:,1], dataset_3.values[:,2])), "x", "y", 1.0, "anscombe.png", test_run, output_folder)

def test_tallo(test_run:bool, output_folder:str):
    # Fetch the classic robust regression star dataset
    trees = fetch_openml(data_id=45081, as_frame=True, parser='auto')
    df = trees.frame

    # Features are light intensity and surface temperature
    if not test_run:
        print(df.head)
        print(df.columns)
        #print(df.values[:,6])

    fit_line_dependent(convert_data((df.values[:,7], df.values[:,6])), "height", "stem diameter", 10.0, "tallo_1.png", test_run, output_folder)

def test_wages(test_run:bool, output_folder:str):
    # Fetch the classic robust regression star dataset
    data_table = fetch_openml(data_id=534, as_frame=True, parser='auto').frame

    # Features are light intensity and surface temperature
    if not test_run:
        print(data_table.head)
        print(data_table.columns)
        #print(df.values[:,6])

    fit_line_dependent(convert_data((data_table.values[:,6], data_table.values[:,3])), "AGE", "EXPERIENCE", 5.0, "wages_1.png", test_run, output_folder)
    
def test_plasma_retinol(test_run:bool, output_folder:str):
    # Fetch the classic robust regression star dataset
    data_table = fetch_openml(data_id=511, as_frame=True, parser='auto').frame

    # Features are light intensity and surface temperature
    #print(data_table.head)
    #print(data_table.columns)

    if False:
        # find correlating factors
        valid_cols = (0,3,5,6,7,8,9,10,11,12,13)
        for i in range(len(valid_cols)):
            vi = valid_cols[i]
            for j in range(i+1,len(valid_cols)):
                vj = valid_cols[j]
                cc = scipy.stats.pearsonr(data_table.values[:,vi].astype(np.double), data_table.values[:,vj].astype(np.double))
                if abs(cc.statistic) > 0.1 and cc.pvalue < 0.1:
                    print("CC:",vi,vj,"cc=",cc.statistic,"pvalue=",cc.pvalue)
                    #fit_line_dependent(convert_data((data_table.values[:,vi], data_table.values[:,vj])), "", "", 50.0, "tallo_1.png", test_run, output_folder)
    
    fit_line_dependent(convert_data((data_table.values[:,5], data_table.values[:,6])), "CALORIES", "FAT", 30.0, "pret_calories_fat.png", test_run, output_folder, fix_origin=True)
    fit_line_dependent(convert_data((data_table.values[:,5], data_table.values[:,11])), "CALORIES", "RETDIET", 500.0, "pret_calories_retdiet_1.png", test_run, output_folder, fix_origin=True)

def test_hertzprung_russell(test_run:bool, output_folder:str):
    data = np.array(sm.datasets.get_rdataset("starsCYG", "robustbase", cache=True).data.values)
    fit_line_orthog("BestAngle", data, "log(Temp)", "log(Light)", 0.2, "hertzprung_russell_bestangle.png", test_run, output_folder, axis_def="fix range")
    fit_line_orthog("IRLS", data, "log(Temp)", "log(Light)", 0.2, "hertzprung_russell_irls.png", test_run, output_folder, axis_def="fix range")

def test_prestige(test_run:bool, output_folder:str):
    prestige = sm.datasets.get_rdataset("Duncan", "carData", cache=True).data
    fit_line_dependent(convert_data((prestige.income.values, prestige.prestige.values)), "Income", "Prestige", 10.0, "prestige_income.png", test_run, output_folder, fix_origin=True)
    fit_line_dependent(convert_data((prestige.education.values, prestige.prestige.values)), "Education", "Prestige", 10.0, "prestige_education.png", test_run, output_folder, fix_origin=True)

def test_stackloss(test_run:bool, output_folder:str):
    data = sm.datasets.stackloss.load()
    data.exog = sm.add_constant(data.exog)
    fit_line_dependent(convert_data((data.endog.values, data.exog.values[:,1])), "endog", "AIRFLOW", 1.0, "stackloss_airflow.png", test_run, output_folder)
    fit_line_dependent(convert_data((data.endog.values, data.exog.values[:,2])), "endog", "WATERTEMP", 1.0, "stackloss_watertemp.png", test_run, output_folder)
    fit_line_dependent(convert_data((data.endog.values, data.exog.values[:,3])), "endog", "ACIDCONC", 1.0, "stackloss_acidconc.png", test_run, output_folder)

def test_statsmodel_1(idx:int, test_run:bool, output_folder:str, do_test:bool):
    np.random.seed(28231*idx) # We want the numbers to be the same on each run

    nsample = 50
    x1 = np.linspace(0, 20, nsample)
    X = np.column_stack((x1, (x1 - 5) ** 2))
    X = sm.add_constant(X)
    sig = 0.3  # smaller error variance makes OLS<->RLM contrast bigger
    beta = [5, 0.5, -0.0]
    y_true2 = np.dot(X, beta)
    y2 = y_true2 + sig * 1.0 * np.random.normal(size=nsample)
    y2[[39, 41, 43, 45, 48]] -= 5  # add some outliers (10% of nsample)
    #y2[[0, 7, 14, 22, 31]] -= 5  # add some outliers (10% of nsample)
    if do_test:
        fit_line_dependent(convert_data((x1, y2)), "x", "y", sig, "statsmodel_1.png", test_run, output_folder)

def main(test_run:bool, output_folder:str="../../../output"):
    output_folder += "/line_fit/examples"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # examples used at https://www.statsmodels.org/stable/examples/notebooks/generated/robust_models_1.html
    #test_hertzprung_russell(test_run, output_folder)
    test_prestige(test_run, output_folder)

    #test_anscombe(test_run, output_folder)
    #test_tallo(test_run, output_folder)
    test_wages(test_run, output_folder)
    test_plasma_retinol(test_run, output_folder)

    # examples used at https://www.statsmodels.org/stable/examples/notebooks/generated/robust_models_0.html
    test_stackloss(test_run, output_folder)
    for i in range(2 if test_run else 20):
        test_statsmodel_1(i, test_run, output_folder, True if i >= 5 else False)

    if test_run:
        print("line_fit_examples OK")

if __name__ == "__main__":
    main(False) # test_run

# Other datasets available from https://www.openml.org/
#Brazilian House dataset: 42688 44062 44047 43999 44016 44152(reproduced) 44990
#balloon: 512
#Tallo trees: 45081
#Wine 43589 40498 44136 44011 43994 43986 287 43351
#cps_85_wages: 534
#plasma_retinol: 511
#strikes: 549
