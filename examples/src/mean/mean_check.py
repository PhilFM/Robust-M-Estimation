import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    import sys
    sys.path.append("../../../pypi_package/src")

from gnc_smoothie.sup_gauss_newton import SupGaussNewton
from gnc_smoothie.irls import IRLS
from gnc_smoothie.draw_functions import gncs_draw_data_points
from gnc_smoothie.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie.gnc_null_params import GNC_NullParams
from gnc_smoothie.gnc_irls_p_params import GNC_IRLSpParams
from gnc_smoothie.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie.pseudo_huber_influence_func import PseudoHuberInfluenceFunc
from gnc_smoothie.gnc_irls_p_influence_func import GNC_IRLSpInfluenceFunc
from gnc_smoothie.plt_alg_vis import gncs_draw_vline, gncs_draw_curve
from gnc_smoothie.linear_model.linear_regressor import LinearRegressor

from flat_welsch_mean import flat_welsch_mean

n = 10
xrange = 10.0
sigma_base = 0.5
sigma_limit = xrange
num_sigma_steps = 20 #50
max_niterations = 100
diff_thres = 1.0e-15

def objective_func(m, optimiser_instance):
    return optimiser_instance.objective_func([m])

def plot_result(data, weight,
               gnc_welsch_optimiser_instance, m_gnc_welsch_irls, m_gnc_welsch_supgn, m_flat,
               pseudo_huber_optimiser_instance, m_pseudo_huber_supgn, m_pseudo_huber_irls,
               gnc_irls_p_optimiser_instance, m_gncirlsp,
               output_folder:str,
               test_run:bool):
    dmin = min(data)
    dmax = max(data)

    # allow border
    drange = dmax-dmin
    x_min = dmin - 0.05*drange
    x_max = dmax + 0.05*drange

    mlist = np.linspace(x_min, x_max, num=300)

    plt.close("all")
    plt.figure(num=1, dpi=240)
    ax = plt.gca()
    rmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
    key = ("Flat", "Welsch", "GNC_Welsch")
    gncs_draw_curve(plt, rmfv(mlist, optimiser_instance=gnc_welsch_optimiser_instance), key, xvalues=mlist, draw_markers=False, hlight_x_value=m_flat, ax=ax)
    gncs_draw_vline(plt, m_flat,       key, use_label=False)
    key = ("SupGN", "Welsch", "GNC_Welsch")
    gncs_draw_curve(plt, rmfv(mlist, optimiser_instance=gnc_welsch_optimiser_instance), key, xvalues=mlist, draw_markers=False, hlight_x_value=m_gnc_welsch_supgn, ax=ax)
    gncs_draw_vline(plt, m_gnc_welsch_supgn,     key, use_label=False)
    gncs_draw_vline(plt, m_gnc_welsch_irls, ("IRLS", "Welsch",      "GNC_Welsch"))
    key = ("SupGN", "PseudoHuber", "Welsch")
    gncs_draw_curve(plt, 0.03*rmfv(mlist, optimiser_instance=pseudo_huber_optimiser_instance), key, xvalues=mlist, draw_markers=False, hlight_x_value=m_pseudo_huber_supgn, ax=ax)
    gncs_draw_vline(plt, m_pseudo_huber_supgn,      key, use_label=False)
    #gncs_draw_vline(plt, m_pseudo_huber_irls, ("IRLS", "PseudoHuber", "Welsch"))
    key = ("IRLS", "GNC_IRLSp", "GNC_IRLSp0")
    gncs_draw_curve(plt, 0.02*rmfv(mlist, optimiser_instance=gnc_irls_p_optimiser_instance), key, xvalues=mlist, draw_markers=False, hlight_x_value=m_gncirlsp, ax=ax)
    gncs_draw_vline(plt, m_gncirlsp,   key, use_label=False)

    gncs_draw_data_points(plt, data, x_min, x_max, n, weight=weight)
    plt.legend()
    plt.savefig(os.path.join(output_folder, "mean_check.png"), bbox_inches='tight')
    if not test_run:
        plt.show()
    
def main(test_run:bool, output_folder:str="../../../output"):
    np.random.seed(0) # We want the numbers to be the same on each run

    for test_idx in range(0,10):
        data = np.zeros((n,1))
        weight = np.ones(n)
        if test_idx == 0:
            for j in range(n):
                if j == 0:
                    data[j] = [0.89]
                elif j < 3:
                    data[j] = [0.9 + 0.01*j]
                else:
                    data[j] = [j*xrange/n]
        else:
            for j in range(n):
                d = np.random.rand()*xrange
                data[j] = [d]

        messages_file = None

        model_instance = LinearRegressor(data[0])

        #print("data(1)=",data)
        param_instance = GNC_WelschParams(WelschInfluenceFunc(), sigma_base,
                                          sigma_limit=sigma_limit, num_sigma_steps=num_sigma_steps)
        irls_instance = IRLS(param_instance, data, model_instance=model_instance, weight=weight,
                             max_niterations=max_niterations, diff_thres=diff_thres, messages_file=messages_file)
        irls_instance.run() # this can fail but let's use the result anyway
        m_gnc_welsch_irls = irls_instance.final_model

        gnc_welsch_optimiser_instance = SupGaussNewton(param_instance, data, model_instance=model_instance, weight=weight,
                                                       max_niterations=max_niterations, diff_thres=diff_thres, messages_file=messages_file)
        if gnc_welsch_optimiser_instance.run():
            m_gnc_welsch_supgn = gnc_welsch_optimiser_instance.final_model

        m_flat = flat_welsch_mean(data, sigma_base, weight,
                                  max_niterations=max_niterations, diff_thres=diff_thres, messages_file=messages_file)

        param_instance = GNC_NullParams(PseudoHuberInfluenceFunc(sigma_base))
        pseudo_huber_optimiser_instance = SupGaussNewton(param_instance, data, model_instance=model_instance, weight=weight,
                                                         max_niterations=max_niterations, diff_thres=diff_thres, messages_file=messages_file)
        if pseudo_huber_optimiser_instance.run():
            m_pseudo_huber_supgn = pseudo_huber_optimiser_instance.final_model

        irls_instance = IRLS(param_instance, data, model_instance=model_instance, weight=weight,
                             max_niterations=max_niterations, diff_thres=diff_thres, messages_file=messages_file)
        if irls_instance.run():
            m_pseudo_huber_irls = irls_instance.final_model

        # GNC IRLS-p params [p,epsilon_base,epsilon_limit,rscale,beta]
        param_instance = GNC_IRLSpParams(GNC_IRLSpInfluenceFunc(), 0.0, 0.01, 1.0,
                                         epsilon_limit=1.0/xrange, beta=0.8)
        irls_instance = IRLS(param_instance, data, model_instance=model_instance, weight=weight,
                             max_niterations=max_niterations, diff_thres=diff_thres, messages_file=messages_file)
        if irls_instance.run():
            m_gncirlsp = irls_instance.final_model

        gnc_irls_p_optimiser_instance = SupGaussNewton(param_instance, data, model_instance=model_instance, weight=weight,
                                                       max_niterations=max_niterations, diff_thres=diff_thres, messages_file=messages_file)

        plot_result(data, weight,
                    gnc_welsch_optimiser_instance,   m_gnc_welsch_irls,    m_gnc_welsch_supgn, m_flat,
                    pseudo_huber_optimiser_instance, m_pseudo_huber_supgn, m_pseudo_huber_irls,
                    gnc_irls_p_optimiser_instance,   m_gncirlsp,
                    output_folder, test_run)

    if test_run:
        print("mean_check OK")

if __name__ == "__main__":
    main(False) # test_run
