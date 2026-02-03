import numpy as np
import matplotlib.pyplot as plt
import os
import math
import pandas as pd

if __name__ == "__main__":
    import sys
    sys.path.append("../../../pypi_package/src")
    
from gnc_smoothie_philfm.plt_alg_vis import gncs_draw_curve, gncs_draw_vline

from mean_leave_out_apply import mean_leave_out_apply, StatsResult, calculate_stats
from weighted_mode import weighted_histogram

def save_histogram(data: np.array,
                   stats_result_ref: StatsResult, 
                   dataset_name: str,
                   dataset_label: str, 
                   sigma_pop: float,
                   add_m_estimator_reference: bool,
                   add_l_estimator_reference: bool,
                   output_folder: str):
    bin_size = 0.6*sigma_pop
    x_min,counts = weighted_histogram(data, bin_size=bin_size)
    plt.close("all")
    plt.figure(num=1, dpi=240)
    plt.clf()
    ax = plt.gca()
    if add_m_estimator_reference:
        gncs_draw_vline(plt, 2.0*(stats_result_ref.m_gnc_welsch   - x_min - 0.5*bin_size)/bin_size,  ("IRLS",   "Welsch",      "GNC_Welsch"), use_label=True)
        gncs_draw_vline(plt, 2.0*(stats_result_ref.m_pseudo_huber - x_min - 0.5*bin_size)/bin_size,  ("IRLS",   "PseudoHuber", "Welsch"    ), use_label=True)
        gncs_draw_vline(plt, 2.0*(stats_result_ref.m_gnc_irls_p   - x_min - 0.5*bin_size)/bin_size,  ("IRLS",   "GNC_IRLSp",   "GNC_IRLSp0"), use_label=True)
        gncs_draw_vline(plt, 2.0*(stats_result_ref.m_rme          - x_min - 0.5*bin_size)/bin_size,  ("RME",    "",            ""          ), use_label=True)
    elif add_l_estimator_reference:
        gncs_draw_vline(plt, 2.0*(stats_result_ref.mean           - x_min - 0.5*bin_size)/bin_size,  ("Mean",   "Basic",       ""          ), use_label=True)
        gncs_draw_vline(plt, 2.0*(stats_result_ref.m_trimmed      - x_min - 0.5*bin_size)/bin_size,  ("Mean",   "Trimmed",     ""          ), use_label=True)
        gncs_draw_vline(plt, 2.0*(stats_result_ref.median         - x_min - 0.5*bin_size)/bin_size,  ("Median", "Basic",       ""          ), use_label=True)
        gncs_draw_vline(plt, 2.0*(stats_result_ref.trimean        - x_min - 0.5*bin_size)/bin_size,  ("Trimean","Basic",       ""          ), use_label=True)

    x_max = x_min + 0.5*(1+len(counts))*bin_size
    ticks = []
    labels = []
    step = 1
    while True:
        if 10*step < x_max-x_min:
            step *= 5
        else:
            break

        if 10*step < x_max-x_min:
            step *= 2
        else:
            break

    xi_min = step*(int(x_min)//step)
    xi_max = step*(1+int(x_max)//step)
    for x in range(xi_min,xi_max,step):
        ticks.append(2.0*(x-x_min)/bin_size)
        labels.append(x)

    ax.set_xticks(ticks, labels=labels)
    ax.set_xlabel(r"Histogram of " + dataset_label)
    ax.set_ylim(0.0,1.05*max(counts))
    
    plt.plot(counts)
    if add_m_estimator_reference or add_l_estimator_reference:
        plt.legend()

        
    if add_m_estimator_reference:
        plt.savefig(os.path.join(output_folder, "leave_out_hist_" + dataset_name + "_mref.png"), bbox_inches='tight')
    elif add_l_estimator_reference:
        plt.savefig(os.path.join(output_folder, "leave_out_hist_" + dataset_name + "_lref.png"), bbox_inches='tight')
    else:
        plt.savefig(os.path.join(output_folder, "leave_out_hist_" + dataset_name + ".png"), bbox_inches='tight')
    #plt.show()

def process_dataset(data, dataset_name: str, dataset_label: str, sigma_pop: float, test_run: bool, output_folder: str, quick_run: bool):
    val_gncwelsch_list = []
    val_mean_list = []
    val_huber_list = []
    val_trimmed_list = []
    val_median_list = []
    val_trimean_list = []
    val_gncirlsp_list = []
    val_rme_list = []
    n_samples = 20 if quick_run else 1000
    leave_out_fraction_list = [0.9,0.92,0.94] if quick_run else [0.95,0.96,0.97,0.98,0.99] #[0.95,0.96,0.97,0.98,0.99]

    welsch_q = 0.62 # 0.5
    pseudo_huber_sigma_scale = 0.6 # 1.2
    gnc_irls_p_epsilon_scale = 1.4 #1.66
    rme_beta_scale = 0.95 #1.4
    
    # calculate stats reference
    stats_result_ref = calculate_stats(data,
                                       sigma_pop,
                                       welsch_q,
                                       pseudo_huber_sigma_scale,
                                       gnc_irls_p_epsilon_scale,
                                       rme_beta_scale,
                                       test_run)

    save_histogram(data, stats_result_ref, dataset_name, dataset_label, sigma_pop, True, False, output_folder)
    save_histogram(data, stats_result_ref, dataset_name, dataset_label, sigma_pop, False, True, output_folder)

    for leave_out_fraction in leave_out_fraction_list:
        if (1.0-leave_out_fraction)*len(data) > 3:
            output_file_1 = None # Path("../../../output/leave_out-" + str(int(x_range)) + "-" + str(n) + "-" + str(int(100.0*leave_out_fraction)) + "_1.png")
            output_file_2 = None # Path("../../../output/leave_out-" + str(int(x_range)) + "-" + str(n) + "-" + str(int(100.0*leave_out_fraction)) + "_2.png")
            alg_result = mean_leave_out_apply(
                data,
                sigma_pop,
                leave_out_fraction,
                n_samples,
                welsch_q,
                pseudo_huber_sigma_scale,
                gnc_irls_p_epsilon_scale,
                rme_beta_scale,
                stats_result_ref,
                output_file_1=output_file_1,
                output_file_2=output_file_2,
                test_run=test_run)

            normaliser = 1.0*math.pow(alg_result.n0, -0.8)

            # GNC Welsch
            val = math.pow(alg_result.sd_gnc_welsch, -1.0)*normaliser
            if not test_run:
                print("GNC Welsch estimator error inverse variance: ", val)

            val_gncwelsch_list.append(val)

            # Arithmetic mean
            val = math.pow(alg_result.sd_mean, -1.0)*normaliser
            if not test_run:
                print("Mean error inverse variance: ", val)

            val_mean_list.append(val)

            # Pseudo-Huber
            val = math.pow(alg_result.sd_huber, -1.0)*normaliser
            if not test_run:
                print("Pseudo-Huber estimator error inverse variance: ", val)

            val_huber_list.append(val)

            # Trimmed mean
            val = math.pow(alg_result.sd_trimmed, -1.0)*normaliser
            if not test_run:
                print("Trimmed mean 50% error inverse variance: ", val)

            val_trimmed_list.append(val)

            # Median
            val = math.pow(alg_result.sd_median, -1.0)*normaliser
            if not test_run:
                print("Median error inverse variance: ", val)

            val_median_list.append(val)

            # Tukey trimean
            val = math.pow(alg_result.sd_trimean, -1.0)*normaliser
            if not test_run:
                print("Tukey trimean error inverse variance: ", val)

            val_trimean_list.append(val)

            # GNC IRLS-p
            val = math.pow(alg_result.sd_gnc_irls_p, -1.0)*normaliser
            if not test_run:
                print("GNC IRLS-p=0 estimator error inverse variance: ", val)

            val_gncirlsp_list.append(val)

            # RME
            val = math.pow(alg_result.sd_rme, -1.0)*normaliser
            if not test_run:
                print("Robust Mean Estimator error inverse variance: ", val)

            val_rme_list.append(val)

    min_val = min(min(val_gncwelsch_list), min(val_mean_list), min(val_huber_list), min(val_trimmed_list), min(val_median_list), min(val_trimean_list), min(val_gncirlsp_list), min(val_rme_list))
    max_val = max(max(val_gncwelsch_list), max(val_mean_list), max(val_huber_list), max(val_trimmed_list), max(val_median_list), max(val_trimean_list), max(val_gncirlsp_list), max(val_rme_list))

    plt.close("all")
    plt.figure(num=1, dpi=240)
    plt.clf()
    ax = plt.gca()
    used_leave_out_fraction_list = leave_out_fraction_list[0:len(val_gncwelsch_list)]
    gncs_draw_curve(plt, val_gncwelsch_list,    ("IRLS",   "Welsch",      "GNC_Welsch"), xvalues=used_leave_out_fraction_list)
    gncs_draw_curve(plt, val_mean_list,         ("Mean",   "Basic",       ""          ), xvalues=used_leave_out_fraction_list)
    gncs_draw_curve(plt, val_trimmed_list,      ("Mean",   "Trimmed",     ""          ), xvalues=used_leave_out_fraction_list)
    gncs_draw_curve(plt, val_median_list,       ("Median", "Basic",       ""          ), xvalues=used_leave_out_fraction_list)
    gncs_draw_curve(plt, val_trimean_list,      ("Trimean","Basic",       ""          ), xvalues=used_leave_out_fraction_list)

    ax.set_xticks(leave_out_fraction_list)

    #print("val_gncwelsch_list",val_gncwelsch_list)
    ax.set_xlabel(r"Leave out fraction")
    ax.set_ylabel("Inverse square-root of MSE")
    #plt.box(False)
    ax.set_xlim(leave_out_fraction_list[0],leave_out_fraction_list[len(leave_out_fraction_list)-1])
    ax.set_ylim(0.95*min_val,1.05*max_val)

    plt.legend()
    plt.savefig(os.path.join(output_folder, "leave_out_" + dataset_name + "_lref.png"), bbox_inches='tight')
    if False: #not test_run:
        plt.show()

    plt.close("all")
    plt.figure(num=1, dpi=240)
    plt.clf()
    ax = plt.gca()
    used_leave_out_fraction_list = leave_out_fraction_list[0:len(val_gncwelsch_list)]
    gncs_draw_curve(plt, val_gncwelsch_list,    ("IRLS",   "Welsch",      "GNC_Welsch"), xvalues=used_leave_out_fraction_list)
    gncs_draw_curve(plt, val_huber_list,        ("IRLS",   "PseudoHuber", "Welsch"    ), xvalues=used_leave_out_fraction_list)
    gncs_draw_curve(plt, val_gncirlsp_list,     ("IRLS",   "GNC_IRLSp",   "GNC_IRLSp0"), xvalues=used_leave_out_fraction_list)
    gncs_draw_curve(plt, val_rme_list,          ("RME",    "",            ""          ), xvalues=used_leave_out_fraction_list)

    ax.set_xticks(leave_out_fraction_list)

    #print("val_gncwelsch_list",val_gncwelsch_list)
    ax.set_xlabel(r"Leave out fraction")
    ax.set_ylabel("Inverse square-root of MSE")
    #plt.box(False)
    ax.set_xlim(leave_out_fraction_list[0],leave_out_fraction_list[len(leave_out_fraction_list)-1])
    ax.set_ylim(0.95*min_val,1.05*max_val)

    plt.legend()
    plt.savefig(os.path.join(output_folder, "leave_out_" + dataset_name + "_mref.png"), bbox_inches='tight')
    if False: #not test_run:
        plt.show()

def main(test_run:bool, output_folder:str="../../../output", quick_run:bool=False):
    np.random.seed(0) # We want the numbers to be the same on each run

    if not test_run:
        train = pd.read_csv("data/loan/Outlier_Loan_dataset.csv")
        cleaned = pd.read_csv("data/loan/Final_Outliers_clean_dataset.csv")

        # get ground-truth list of outliers
        id = list(zip(train["User_ID"],train["Loan_ID"]))
        id_set = set(id)
        id_cleaned = list(zip(cleaned["User_ID"],cleaned["Loan_ID"]))
        id_cleaned_set = set(id_cleaned)
        #print("id_cleaned len ",len(id_cleaned), len(id_cleaned_set))

        id_outlier = id_set.difference(id_cleaned_set)
        #print("id_outlier:",id_outlier)
        print("tot ",len(id)," clean ", len(id_cleaned), " outlier ",len(id_outlier))

        column = train["Loan_Amount"]
        fil_val = []
        for c in column:
            if not math.isnan(c):
                fil_val.append(c)

        process_dataset(np.array(fil_val).reshape((len(fil_val),1)), "Loan_Amount", "loan amount", 40000.0, test_run, output_folder, quick_run)

        column = train["Current_Balance"]
        fil_val = []
        for c in column:
            if not math.isnan(c):
                fil_val.append(c)

        #print("Current_Balance:",fil_val)
        process_dataset(np.array(fil_val).reshape((len(fil_val),1)), "Current_Balance", "current balance", 30000.0, test_run, output_folder, quick_run)

        column = train["Account_Balance"]
        fil_val = []
        for c in column:
            if not math.isnan(c):
                fil_val.append(c)

        #print("Account_Balance:",fil_val)
        process_dataset(np.array(fil_val).reshape((len(fil_val),1)), "Account_Balance", "account balance", 50000.0, test_run, output_folder, quick_run)

        column = train["Credit_Score"]
        process_dataset(np.array(column).reshape((len(column),1)), "Credit_Score", "credit score", 50.0, test_run, output_folder, quick_run)
        
    # simulated data
    np.random.seed(10) # We want the numbers to be the same on each run
    if False:
        for idx in range(4):
            sigma_pop = 2.0
            x_gt_border = 0.0 #3.0*sigma_pop
            x_range = 10.0
            m_gt = np.random.rand()*x_range + x_gt_border
            n_data = 1000
            data = np.zeros((n_data,1))
            outlier_fraction = 0.3
            n0 = int((1.0-outlier_fraction)*n_data+0.5)
            for j in range(n_data):
                data[j] = [np.random.normal(loc=m_gt, scale=sigma_pop)]

            for j in range(n_data-n0):
                data[n0+j] = [np.random.rand()*(x_range + 2.0*x_gt_border)]

            process_dataset(data, "simulated-" + str(1+idx), "simulated data", sigma_pop, test_run, output_folder, quick_run)

    if test_run:
        print("mean_leave_out OK")

if __name__ == "__main__":
    main(False) # test_run
