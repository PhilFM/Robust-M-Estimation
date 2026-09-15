import numpy as np
import os
import json
import matplotlib.pyplot as plt
from pathlib import Path
import re

def float_from_name(name:str):
    all_ints = re.findall(r"\d+\.\d+", name)
    return float(all_ints[0])

def process_theta_vals(qval, data_model_name, strategy_name, sup_gn_q, test_run:bool, output_folder:str=None):
    theta_vals = []
    alpha_vals = []
    beta_vals = []
    gamma_vals = []
    delta_vals = []
    f_val_vals = []
    for theta_name,tval in qval["theta_vals"].items():
        theta_vals.append(float_from_name(theta_name))
        alpha_vals.append(tval["alpha"])
        beta_vals.append(tval["beta"])
        gamma_vals.append(tval["gamma"])
        delta_vals.append(tval["delta"])
        f_val_vals.append(tval["f_val"])

    plt.close("all")
    plt.figure(num=1, dpi=240)
    plt.plot(theta_vals, alpha_vals, lw = 1.0, label="$\\alpha$")
    plt.plot(theta_vals, beta_vals, lw = 1.0, label="$\\beta$")
    plt.plot(theta_vals, gamma_vals, lw = 1.0, label="$\\gamma$")
    plt.plot(theta_vals, delta_vals, lw = 1.0, label="$\\delta$")
    plt.plot(theta_vals, f_val_vals, lw = 1.0, label="$F(.)$")
    plt.legend()
    plt.savefig(os.path.join(output_folder, "theta_graph_"+data_model_name+"_"+strategy_name+"_q="+str(sup_gn_q)+".png"), bbox_inches='tight')
    if not test_run:
        plt.show()

def process_data_model_strategy(data_model_name:str, strategy_name:str, line_fit_dict, test_run:bool, output_folder:str=None):
    if not test_run:
        print("data_model=",data_model_name,"strategy=",strategy_name)

    q_vals = []
    theta_vals = []
    alpha_vals = []
    beta_vals = []
    gamma_vals = []
    delta_vals = []
    f_val_vals = []
    for sup_gn_q_name,qval in line_fit_dict.items():
        sup_gn_q = float_from_name(sup_gn_q_name)
        #print("sup_gn_q_name=",sup_gn_q_name)
        #print("q=",sup_gn_q)

        q_vals.append(sup_gn_q)
        dsval = qval[data_model_name][strategy_name]
        theta_vals.append(dsval["theta"])
        alpha_vals.append(dsval["alpha"])
        beta_vals.append(dsval["beta"])
        gamma_vals.append(dsval["gamma"])
        delta_vals.append(dsval["delta"])
        f_val_vals.append(dsval["f_val"])

        process_theta_vals(dsval, data_model_name, strategy_name, sup_gn_q, test_run, output_folder)

    if not test_run:
        print("q_vals",q_vals)
        print("theta_vals",theta_vals)
        print("alpha_vals",alpha_vals)

    plt.close("all")
    plt.figure(num=1, dpi=240)
    plt.plot(q_vals, alpha_vals, lw = 1.0, label="$\\alpha$", marker="o")
    ax = plt.gca()
    ax.set_xlabel(r"q ("+data_model_name+" model, "+strategy_name+" strategy)" )
    ax.set_ylabel("Breakdown point $\\alpha$")
    ax.set_ylim(0.0,1.05*max(alpha_vals))

    #plt.legend()
    plt.savefig(os.path.join(output_folder, "alpha_graph_"+data_model_name+"_"+strategy_name+".png"), bbox_inches='tight')
    if not test_run:
        plt.show()

    plt.close("all")
    plt.figure(num=1, dpi=240)
    plt.plot(q_vals, beta_vals, lw = 1.0, label="$\\beta$", marker="o")
    ax = plt.gca()
    ax.set_xlabel(r"q ("+data_model_name+" model, "+strategy_name+" strategy)" )
    ax.set_ylabel("Breakdown point $\\beta$")
    ax.set_ylim(0.0,1.05*max(beta_vals))
    
    #plt.legend()
    plt.savefig(os.path.join(output_folder, "beta_graph_"+data_model_name+"_"+strategy_name+".png"), bbox_inches='tight')
    if not test_run:
        plt.show()

    plt.close("all")
    plt.figure(num=1, dpi=240)
    plt.plot(q_vals, gamma_vals, lw = 1.0, label="$\\gamma$", marker="o")
    ax = plt.gca()
    ax.set_xlabel(r"q ("+data_model_name+" model, "+strategy_name+" strategy)" )
    ax.set_ylabel("Breakdown point $\\gamma$")
    ax.set_ylim(0.0,1.05*max(gamma_vals))
    
    #plt.legend()
    plt.savefig(os.path.join(output_folder, "gamma_graph_"+data_model_name+"_"+strategy_name+".png"), bbox_inches='tight')
    if not test_run:
        plt.show()

    plt.close("all")
    plt.figure(num=1, dpi=240)
    plt.plot(q_vals, delta_vals, lw = 1.0, label="$\\delta$", marker="o")
    ax = plt.gca()
    ax.set_xlabel(r"q ("+data_model_name+" model, "+strategy_name+" strategy)" )
    ax.set_ylabel("Breakdown point $\\delta$")
    #ax.set_ylim(0.0,1.05*max(delta_vals))
    
    #plt.legend()
    plt.savefig(os.path.join(output_folder, "delta_graph_"+data_model_name+"_"+strategy_name+".png"), bbox_inches='tight')
    if not test_run:
        plt.show()

    plt.close("all")
    plt.figure(num=1, dpi=240)
    plt.plot(q_vals, f_val_vals, lw = 1.0, label="$F(.)$", marker="o")
    ax = plt.gca()
    ax.set_xlabel(r"q ("+data_model_name+" model, "+strategy_name+" strategy)" )
    ax.set_ylabel("Breakdown point $F(.)$")
    ax.set_ylim(0.0,1.05*max(f_val_vals))
    
    #plt.legend()
    plt.savefig(os.path.join(output_folder, "f_val_graph_"+data_model_name+"_"+strategy_name+".png"), bbox_inches='tight')
    if not test_run:
        plt.show()

def main(test_run:bool, output_folder:str="../../../output", quick_run:bool=False):
    output_folder += "/line_fit/breakdown_point"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    np.random.seed(0) # We want the numbers to be the same on each run

    # read json data defining location of limiting line parameters in each direction
    try:
        json_file = open("line_fit_breakdown.json")
    except FileNotFoundError: # when running run_all.py
        json_file = open("line_fitting/line_fit_breakdown.json")

    json_str = json_file.read()
    line_fit_dict = json.loads(json_str)
    #print("line_fit_dict=",line_fit_dict)

    data_model_names = []
    strategy_names = []
    
    # get lists of data model names and strategy namesfrom first q value entry
    for key,itm in line_fit_dict.items():
        for k2,it in itm.items():
            data_model_names.append(k2)
            if len(strategy_names) == 0:
                for k3,itp in it.items():
                    strategy_names.append(k3)

        break

    if not test_run:
        print("data_model_names=",data_model_names)
        print("strategy_names=",strategy_names)

    for data_model_name in data_model_names:
        for strategy_name in strategy_names:
            process_data_model_strategy(data_model_name, strategy_name, line_fit_dict, test_run, output_folder)

    if test_run:
        print("line_fit_analyse_breakdown OK")

if __name__ == "__main__":
    main(False) # test_run
