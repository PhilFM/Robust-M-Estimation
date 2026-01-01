import numpy as np
import matplotlib.pyplot as plt
import math

from robust_mean import M_estimator

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.irls import IRLS
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.gnc_null_params import GNC_NullParams
from gnc_smoothie_philfm.gnc_irls_p_params import GNC_IRLSpParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie_philfm.pseudo_huber_influence_func import PseudoHuberInfluenceFunc
from gnc_smoothie_philfm.gnc_irls_p_influence_func import GNC_IRLSpInfluenceFunc
from gnc_smoothie_philfm.draw_functions import gncs_draw_data_points
from gnc_smoothie_philfm.plt_alg_vis import gncs_draw_vline, gncs_draw_curve

from trimmed_mean import trimmed_mean
from gncs_robust_mean import RobustMean
from weighted_mean import weighted_mean

show_others = True

def objective_func(m, optimiser_instance):
    return optimiser_instance.objective_func([m])

def mean_compare_apply(sigma_pop,p,xgtrange,n,n_samples_base,min_n_samples,outlier_fraction, student_t_dof = 0,
                       output_file = '', test_run:bool=False, smoothie:bool=False):
    vec_gnc_welsch = vec_mean = vec_pseudo_huber = vec_trimmed = vec_median = vec_gnc_irls_p = vec_rme = 0.0

    n0 = int((1.0-outlier_fraction)*n+0.5)
    n_samples = max(n_samples_base//n, min_n_samples)
    if not test_run:
        print("outlier_fraction=",outlier_fraction," n=",n," n0=",n0," n_samples=",n_samples," student_t_dof=",student_t_dof)

    x_gt_border = 3.0*sigma_pop
    data_array = []
    for i in range(n_samples):
        m_gt = np.random.rand()*xgtrange + x_gt_border
        data = np.zeros((n,1))
        weight = np.zeros(n)
        good_data = []
        for j in range(n0):
            if student_t_dof > 1:
                d = m_gt + np.random.standard_t(student_t_dof)
            else:
                d = np.random.normal(loc=m_gt, scale=sigma_pop)

            weight[j] = 1.0
            data[j] = [d]
            good_data.append([weight[j], [d]])

        outlier_data = []
        for j in range(n-n0):
            d = np.random.rand()*(xgtrange + 2.0*x_gt_border)
            weight[n0+j] = 1.0
            data[n0+j] = [d]
            outlier_data.append([weight[n0+j], [d]])

        datap = {}
        datap["good"] = good_data
        datap["outlier"] = outlier_data

        data_array.append(datap)

        model_instance = RobustMean()

        gnc_welsch_sigma = sigma_pop/p
        welsch_irls_instance = IRLS(GNC_WelschParams(WelschInfluenceFunc(), gnc_welsch_sigma, max(xgtrange,10.0*sigma_pop), 100),
                                    model_instance, data, weight=weight, max_niterations=200)

        welsch_supgn_instance = SupGaussNewton(GNC_WelschParams(WelschInfluenceFunc(), gnc_welsch_sigma, max(xgtrange,10.0*sigma_pop), 100),
                                               model_instance, data, weight=weight, max_niterations=200)
        if smoothie:
            # for the smoothie paper use Sup-GN
            if welsch_supgn_instance.run():
                m_gncwelsch = welsch_supgn_instance.final_model
        else:
            # for the mean estimation paper let's use IRLS
            if welsch_irls_instance.run():
                m_gncwelsch = welsch_irls_instance.final_model

        vec_gnc_welsch += math.pow(m_gncwelsch-m_gt, 2.0)

        if not smoothie:
            mean = weighted_mean(data, weight)
            vec_mean += math.pow(mean[0]-m_gt, 2.0)

            pseudo_huber_sigma = sigma_pop/p
            pseudo_huber_irls_instance = IRLS(GNC_NullParams(PseudoHuberInfluenceFunc(sigma=pseudo_huber_sigma)),
                                              model_instance, data, weight=weight)
            pseudo_huber_irls_instance.run()  # this can fail but let's use the result anyway
            m_pseudo_huber = pseudo_huber_irls_instance.final_model

            vec_pseudo_huber += math.pow(m_pseudo_huber-m_gt, 2.0)
            pseudo_huber_supgn_instance = SupGaussNewton(GNC_NullParams(PseudoHuberInfluenceFunc(sigma=pseudo_huber_sigma)),
                                                         model_instance, data, weight=weight)

            #trim_size = n//10
            #m_trimmed = trimmed_mean(data, trim_size=trim_size)
            #vec_trimmed1 += math.pow(m_trimmed-m_gt, 2.0)

        if student_t_dof == 0:
            trim_size = int(0.5 + 0.5*outlier_fraction*n)
        else:
            trim_size = n//4

        m_trimmed = trimmed_mean(data, weight, trim_size=trim_size)
        vec_trimmed += math.pow(m_trimmed-m_gt, 2.0)

        median = np.median(data)
        vec_median += math.pow(median-m_gt, 2.0)

        if not smoothie:
            gnc_irls_p_rscale = 1.0/xgtrange
            gnc_irls_p_epsilon_base = gnc_irls_p_rscale*sigma_pop
            gnc_irls_p_epsilon_limit = 1.0
            gnc_irls_p_p = 0.0
            gnc_irls_p_beta = 0.8
            gnc_irls_p_instance = IRLS(GNC_IRLSpParams(GNC_IRLSpInfluenceFunc(),
                                                       gnc_irls_p_p, gnc_irls_p_rscale, gnc_irls_p_epsilon_base, gnc_irls_p_epsilon_limit, gnc_irls_p_beta),
                                       model_instance, data, weight=weight)
            gnc_irls_p_instance.run() # this can fail but let's use the result anyway
            m_gnc_irls_p = gnc_irls_p_instance.final_model

            vec_gnc_irls_p += math.pow(m_gnc_irls_p-m_gt, 2.0)
            gnc_irlsp_supgn_instance = SupGaussNewton(GNC_IRLSpParams(GNC_IRLSpInfluenceFunc(),
                                                                      gnc_irls_p_p, gnc_irls_p_rscale, gnc_irls_p_epsilon_base,
                                                                      gnc_irls_p_epsilon_limit, gnc_irls_p_beta),
                                                      model_instance, data, weight=weight)

            m_rme = M_estimator(data, beta=1)
            vec_rme += math.pow(m_rme-m_gt, 2.0)

        if len(output_file) > 0:
            # get min and max of data
            y_min = y_max = 0.0
            x_min = x_max = None
            # override x limit 
            #x_min = 0.7
            #x_max = 1.3

            if x_min is None:
                dmin = dmax = data[0]
                for d in data:
                    dmin = min(dmin, d)
                    dmax = max(dmax, d)

                # allow border
                drange = dmax-dmin
                x_min = dmin - 0.05*drange
                x_max = dmax + 0.05*drange

            mlist = np.linspace(x_min, x_max, num=300)

            for mx in mlist:
                y_max = max(y_max, objective_func(mx, welsch_supgn_instance))

            y_min *= 1.1 # allow for a small border
            y_max *= 1.1 # allow for a small border            

            plt.figure(num=1, dpi=240)
            ax = plt.gca()
            #plt.box(False)
            ax.set_ylim((y_min, y_max))

            rmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
            if smoothie:
                gncs_draw_curve(plt, rmfv(mlist, optimiser_instance=welsch_supgn_instance), ("SupGN", "Welsch", "GNC_Welsch"), xvalues=mlist, draw_markers=False, hlight_x_value=m_gncwelsch, ax=ax)
            else:
                gncs_draw_curve(plt, rmfv(mlist, optimiser_instance=welsch_irls_instance), ("IRLS", "Welsch", "GNC_Welsch"), xvalues=mlist, draw_markers=False, hlight_x_value=m_gncwelsch, ax=ax)

                hmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
                hmfv_scaled = hmfv(mlist, optimiser_instance=pseudo_huber_supgn_instance)
                hmfv_scaled *= 0.5
                gncs_draw_curve(plt, hmfv_scaled, ("IRLS", "PseudoHuber", "Welsch"), xvalues=mlist, draw_markers=False, hlight_x_value=m_pseudo_huber, ax=ax)

                gmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
                gmfv_scaled = gmfv(mlist, optimiser_instance=gnc_irlsp_supgn_instance)
                gmfv_scaled *= 0.1
                gncs_draw_curve(plt, gmfv_scaled, ("IRLS", "GNC_IRLSp", "GNC_IRLSp0"), xvalues=mlist, draw_markers=False, hlight_x_value=m_gnc_irls_p, ax=ax)

            gncs_draw_vline(plt, m_gt, ("GroundTruth", "", ""))
            if smoothie:
                gncs_draw_vline(plt, m_gncwelsch, ("SupGN", "Welsch", "GNC_Welsch"), use_label=False)
            else:
                gncs_draw_vline(plt, m_gncwelsch, ("IRLS", "Welsch", "GNC_Welsch"), use_label=False)

            if show_others:
                if not smoothie:
                    gncs_draw_vline(plt, mean,           ("Mean",   "Basic",       ""          ))
                    gncs_draw_vline(plt, m_pseudo_huber, ("IRLS",   "PseudoHuber", "Welsch"    ), use_label=False)

                gncs_draw_vline(plt, m_trimmed,      ("Mean",   "Trimmed",     ""          ))
                gncs_draw_vline(plt, median,         ("Median", "Basic",       ""          ))
                if not smoothie:
                    gncs_draw_vline(plt, m_gnc_irls_p,   ("IRLS",   "GNC_IRLSp",   "GNC_IRLSp0"), use_label=False)
                    gncs_draw_vline(plt, m_rme,          ("RME",    "",            ""          ))

            gncs_draw_data_points(plt, data, weight, x_min, x_max, n0)

            plt.legend()
            plt.savefig(output_file, bbox_inches='tight')
            plt.show()            

    norm = 1.0/(n_samples-1)
    if smoothie:
        return data_array,m_gt,math.sqrt(vec_gnc_welsch*norm),math.sqrt(vec_trimmed*norm),math.sqrt(vec_median*norm),n_samples
    else:
        return data_array,m_gt,math.sqrt(vec_gnc_welsch*norm),math.sqrt(vec_mean*norm),math.sqrt(vec_pseudo_huber*norm),math.sqrt(vec_trimmed*norm),math.sqrt(vec_median*norm),math.sqrt(vec_gnc_irls_p*norm),math.sqrt(vec_rme*norm),n_samples

