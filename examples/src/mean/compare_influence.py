import math
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    import sys
    sys.path.append("../../../pypi_package/src")

from gnc_smoothie.sup_gauss_newton import SupGaussNewton
from gnc_smoothie.irls import IRLS
from gnc_smoothie.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie.gnc_null_params import GNC_NullParams
from gnc_smoothie.gnc_irls_p_params import GNC_IRLSpParams
from gnc_smoothie.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie.pseudo_huber_influence_func import PseudoHuberInfluenceFunc
from gnc_smoothie.gnc_irls_p_influence_func import GNC_IRLSpInfluenceFunc
from gnc_smoothie.draw_functions import gncs_draw_data_points
from gnc_smoothie.plt_alg_vis import gncs_draw_vline, gncs_draw_curve
from gnc_smoothie.linear_model.linear_regressor import LinearRegressor

n = 10
xgtrange = 10.0
sigma_pop = 1.0
num_sigma_steps = 50
max_niterations = 100
residual_tolerance = 1.e-8
lambda_start = 1.0
lambda_scale = 2.0 #1.0
diff_thres = 1.0e-12
welsch_p = 0.66666667
outlier_fraction = 0.5

def objective_func(m, optimiser_instance):
    return optimiser_instance.objective_func([m])

def plot_result(data, weight,
                m_welschopt,    welsch_optimiser_instance,
                m_pseudo_huber, pseudo_huber_supgn_optimiser_instance,
                m_gnc_irls_p,   gnc_irls_p_optimiser_instance,
                m_gt,
                test_run:bool,
                output_folder:str):
    dmin = min(data)
    dmax = max(data)

    # allow border
    drange = dmax-dmin
    x_min = dmin - 0.05*drange
    x_max = dmax + 0.05*drange

    mlist = np.linspace(x_min, x_max, num=300)

    plt.close("all")
    plt.figure(num=1, dpi=240)
    gncs_draw_data_points(plt, data, x_min, x_max, len(data), weight=weight, scale=0.05)
    if m_gt is not None:
        gncs_draw_vline(plt, m_gt, ("GroundTruth","",""))

    ax = plt.gca()

    rmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
    key = ("SupGN", "Welsch", "GNC_Welsch")
    gncs_draw_curve(plt, rmfv(mlist, optimiser_instance=welsch_optimiser_instance), key, xvalues=mlist, draw_markers=False, hlight_x_value=m_welschopt, ax=ax)
    gncs_draw_vline(plt, m_welschopt, key, use_label=False)

    rmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
    key = ("SupGN", "PseudoHuber", "Welsch")
    gncs_draw_curve(plt, 0.05*rmfv(mlist, optimiser_instance=pseudo_huber_supgn_optimiser_instance), key, xvalues=mlist, draw_markers=False, hlight_x_value=m_pseudo_huber, ax=ax)
    gncs_draw_vline(plt, m_pseudo_huber, key, use_label=False)

    rmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
    key = ("IRLS", "GNC_IRLSp", "GNC_IRLSp0")
    gncs_draw_curve(plt, 0.003*rmfv(mlist, optimiser_instance=gnc_irls_p_optimiser_instance), key, xvalues=mlist, draw_markers=False, hlight_x_value=m_gnc_irls_p, ax=ax)
    gncs_draw_vline(plt, m_gnc_irls_p, key, use_label=False)

    plt.legend()
    plt.savefig(os.path.join(output_folder, "compare_influence.png"), bbox_inches='tight')
    if not test_run:
        plt.show()

def main(test_run:bool, output_folder:str="../../output"):
    np.random.seed(0) # We want the numbers to be the same on each run

    for test_idx in range(0,10):
        n0 = int((1.0-outlier_fraction)*n+0.5)
        sigma_pop = 1.0
        xgtborder = 3.0*sigma_pop
        m_gt = np.random.rand()*xgtrange + xgtborder
        data = np.zeros((n,1))
        weight = np.ones(n)
        for j in range(n0):
            data[j] = [np.random.normal(loc=m_gt, scale=sigma_pop)]

        for j in range(n-n0):
            data[n0+j] = [np.random.rand()*(xgtrange + 2.0*xgtborder)]

        messages_file = None

        if not test_run:
            print("test_idx=",test_idx,"m_gt=",m_gt)

        sigma_base = sigma_pop/welsch_p
        sigma_limit = xgtrange

        model_instance = LinearRegressor(data[0])

        welsch_optimiser_instance = SupGaussNewton(GNC_WelschParams(WelschInfluenceFunc(), sigma_base,
                                                                    sigma_limit=sigma_limit, num_sigma_steps=num_sigma_steps),
                                                   data, model_instance=model_instance, weight=weight,
                                                   max_niterations=max_niterations, residual_tolerance=residual_tolerance,
                                                   lambda_start=lambda_start, lambda_scale=lambda_scale, diff_thres=diff_thres,
                                                   messages_file=messages_file)
        if welsch_optimiser_instance.run():
            m_welsch = welsch_optimiser_instance.final_model
            if not test_run:
                print("m_welsch-m_gt=",m_welsch-m_gt)

        pseudo_huber_supgn_optimiser_instance = IRLS(GNC_NullParams(PseudoHuberInfluenceFunc(sigma=sigma_base)), data,
                                                     model_instance=model_instance, weight=weight,
                                                     max_niterations=max_niterations, diff_thres=diff_thres, messages_file=messages_file)
        if pseudo_huber_supgn_optimiser_instance.run():
            m_pseudo_huber = pseudo_huber_supgn_optimiser_instance.final_model
            if not test_run:
                print("m_pseudo_huber-m_gt=",m_pseudo_huber-m_gt)

        pseudo_huber_supgn_optimiser_instance = SupGaussNewton(GNC_NullParams(PseudoHuberInfluenceFunc(sigma=sigma_base)),
                                                               data, model_instance=model_instance, weight=weight,
                                                               max_niterations=max_niterations, diff_thres=diff_thres, messages_file=messages_file)
    
        gnc_irls_p_p = 0.0
        gnc_irls_p_rscale = 1.0/xgtrange
        gnc_irls_p_epsilon_base = gnc_irls_p_rscale*sigma_base
        gnc_irls_p_epsilon_limit = gnc_irls_p_rscale*sigma_limit
        gnc_irls_p_beta = math.exp((math.log(sigma_base) - math.log(sigma_limit))/(num_sigma_steps - 1.0))
        gnc_irls_p_param_instance = GNC_IRLSpParams(GNC_IRLSpInfluenceFunc(),
                                                    gnc_irls_p_p, gnc_irls_p_rscale, gnc_irls_p_epsilon_base,
                                                    epsilon_limit=gnc_irls_p_epsilon_limit, beta=gnc_irls_p_beta)
        gnc_irls_p_optimiser_instance = IRLS(gnc_irls_p_param_instance, data, model_instance=model_instance, weight=weight,
                                             max_niterations=max_niterations, diff_thres=diff_thres, messages_file=messages_file)
        if gnc_irls_p_optimiser_instance.run():
            m_gnc_irls_p = gnc_irls_p_optimiser_instance.final_model
            if not test_run:
                print("m_gnc_irls_p-m_gt=",m_gnc_irls_p-m_gt)

        gnc_irls_p_supgn_optimiser_instance = SupGaussNewton(gnc_irls_p_param_instance, data, model_instance=model_instance, weight=weight,
                                                             max_niterations=max_niterations, residual_tolerance=residual_tolerance,
                                                             lambda_start=lambda_start, lambda_scale=lambda_scale, diff_thres=diff_thres,
                                                             messages_file=messages_file)
        if gnc_irls_p_supgn_optimiser_instance.run():
            m_gnc_irls_popt = gnc_irls_p_supgn_optimiser_instance.final_model
            if not test_run:
                print("m_gnc_irls_popt-m_gt=",m_gnc_irls_popt-m_gt)

        plot_result(data, weight,
                    m_welsch,       welsch_optimiser_instance,
                    m_pseudo_huber, pseudo_huber_supgn_optimiser_instance,
                    m_gnc_irls_p,   gnc_irls_p_supgn_optimiser_instance,
                    m_gt,
                    test_run,
                    output_folder)

    if test_run:
        print("compare_influence OK")

if __name__ == "__main__":
    main(False) # test_run
