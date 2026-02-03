import numpy as np
import matplotlib.pyplot as plt
import math
import time
import os
import sys
from pathlib import Path

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
from gnc_smoothie_philfm.cython_files.linear_regressor_welsch_evaluator import LinearRegressorWelschEvaluator
from gnc_smoothie_philfm.cython_files.linear_regressor_pseudo_huber_evaluator import LinearRegressorPseudoHuberEvaluator
from gnc_smoothie_philfm.cython_files.linear_regressor_gnc_irls_p_evaluator import LinearRegressorGNC_IRLSpEvaluator

from trimmed_mean import trimmed_mean
from tukey_trimean import tukey_trimean
from save_sample import save_sample

show_others = True

def objective_func(m, optimiser_instance):
    return optimiser_instance.objective_func([m])

class CompareMeanAlgResult:
    # standard deviations
    sd_gnc_welsch: float = None
    sd_mean: float = None
    sd_huber: float = None
    sd_trimmed: float = None
    sd_median: float = None
    sd_trimean: float = None
    sd_gnc_irls_p: float = None
    sd_rme: float = None

    # timing
    time_gnc_welsch: float = 0.0
    time_mean: float = 0.0
    time_huber: float = 0.0
    time_trimmed: float = 0.0
    time_median: float = 0.0
    time_trimean: float = 0.0
    time_gnc_irls_p: float = 0.0
    time_rme: float = 0.0
    
    n_samples: int = None

def mean_compare_apply(sigma_pop: float,
                       xgtrange: float,
                       n: int,
                       n_samples_base: int,
                       min_n_samples: int,
                       outlier_fraction: float,
                       welsch_q: float,
                       pseudo_huber_sigma_scale: float,
                       gnc_irls_p_epsilon_scale: float,
                       rme_beta_scale: float,
                       student_t_dof: int = 0,
                       output_file_1:Path = None,
                       output_file_2:Path = None,
                       test_run:bool=False,
                       output_folder:str = None,
                       smoothie:bool=False) -> CompareMeanAlgResult:
    vec_gnc_welsch = vec_mean = vec_pseudo_huber = vec_trimmed = vec_median = vec_trimean = vec_gnc_irls_p = vec_rme = 0.0
    
    n0 = int((1.0-outlier_fraction)*n+0.5)
    alg_result = CompareMeanAlgResult()
    alg_result.n_samples = max(n_samples_base//n, min_n_samples)
    if not test_run:
        print("outlier_fraction=",outlier_fraction," n=",n," n0=",n0," n_samples=",alg_result.n_samples," student_t_dof=",student_t_dof)

    x_gt_border = 0.0 #3.0*sigma_pop
    for sample in range(alg_result.n_samples):
        alg_result.m_gt = np.random.rand()*xgtrange + x_gt_border
        data = np.zeros((n,1))
        weight = np.ones(n)
        good_data = []
        for j in range(n0):
            if student_t_dof > 0:
                d = alg_result.m_gt + np.random.standard_t(student_t_dof)
            else:
                d = np.random.normal(loc=alg_result.m_gt, scale=sigma_pop)

            data[j] = [d]
            good_data.append([weight[j], [d]])

        outlier_data = []
        for j in range(n-n0):
            d = np.random.rand()*(xgtrange + 2.0*x_gt_border)
            data[n0+j] = [d]
            outlier_data.append([weight[n0+j], [d]])

        if sample < 10:
            save_sample(data, sigma_pop, output_folder,
                        "compare_sample_n" + str(n) + "_range" + str(int(xgtrange)) + "_of" + str(int(100.0*outlier_fraction)) + "_sd" + str(student_t_dof) + "_" + str(sample+1) + ".png")

        evaluator_instance = LinearRegressorWelschEvaluator(data[0])

        gnc_welsch_sigma = sigma_pop/welsch_q
        if smoothie:
            # for the smoothie paper use Sup-GN
            welsch_supgn_instance = SupGaussNewton(
                GNC_WelschParams(
                    WelschInfluenceFunc(),
                    gnc_welsch_sigma,
                    sigma_limit=max(max(data)-min(data),xgtrange,10.0*sigma_pop),
                    num_sigma_steps=10),
                data,
                evaluator_instance=evaluator_instance,
                weight=weight,
                max_niterations=200)
            start_time = time.time()
            if welsch_supgn_instance.run():
                m_gnc_welsch = welsch_supgn_instance.final_model

            alg_result.time_gnc_welsch += time.time()-start_time
        else:
            # for the mean estimation paper let's use IRLS
            welsch_irls_instance = IRLS(
                GNC_WelschParams(
                    WelschInfluenceFunc(),
                    gnc_welsch_sigma,
                    sigma_limit=max(max(data)-min(data),xgtrange,10.0*sigma_pop),
                    num_sigma_steps=10),
                data,
                evaluator_instance=evaluator_instance,
                weight=weight,
                max_niterations=200)
            start_time = time.time()
            if welsch_irls_instance.run():
                m_gnc_welsch = welsch_irls_instance.final_model

            alg_result.time_gnc_welsch += time.time()-start_time

        vec_gnc_welsch += math.pow(m_gnc_welsch-alg_result.m_gt, 2.0)

        if not smoothie:
            mean = evaluator_instance.weighted_fit([data], [weight])
            vec_mean += math.pow(mean[0]-alg_result.m_gt, 2.0)

            evaluator_instance = LinearRegressorPseudoHuberEvaluator(data[0])

            pseudo_huber_sigma = sigma_pop*pseudo_huber_sigma_scale
            pseudo_huber_irls_instance = IRLS(GNC_NullParams(PseudoHuberInfluenceFunc(sigma=pseudo_huber_sigma)),
                                              data, evaluator_instance=evaluator_instance, weight=weight)
            start_time = time.time()
            pseudo_huber_irls_instance.run()  # this can fail but let's use the result anyway
            alg_result.time_huber += time.time()-start_time
            m_pseudo_huber = pseudo_huber_irls_instance.final_model

            vec_pseudo_huber += math.pow(m_pseudo_huber-alg_result.m_gt, 2.0)
            pseudo_huber_supgn_instance = SupGaussNewton(GNC_NullParams(PseudoHuberInfluenceFunc(sigma=pseudo_huber_sigma)),
                                                         data, evaluator_instance=evaluator_instance, weight=weight)

        if student_t_dof == 0:
            # correct trim_size to match level of outliers would be 0.5*outlier_fraction*n, but this assumes that the outliers
            # are evenly distributed above and below the good data. To allow for all the outliers to be below (or above) the
            # good data would require a trim size of outlier_fraction*n, but this is overly pessimistic.
            # So let's compromise with 0.75.
            trim_size = int(0.5 + 0.75*outlier_fraction*n)
        else:
            trim_size = n//4

        start_time = time.time()
        m_trimmed = trimmed_mean(data, trim_size=trim_size, weight=weight)
        alg_result.time_trimmed += time.time()-start_time
        vec_trimmed += math.pow(m_trimmed-alg_result.m_gt, 2.0)

        start_time = time.time()
        median = np.median(data)
        alg_result.time_median += time.time()-start_time
        vec_median += math.pow(median-alg_result.m_gt, 2.0)

        start_time = time.time()
        trimean = tukey_trimean(data)
        alg_result.time_trimean += time.time()-start_time
        vec_trimean += math.pow(trimean-alg_result.m_gt, 2.0)

        if not smoothie:
            gnc_irls_p_rscale = 1.0/xgtrange
            gnc_irls_p_epsilon_base = gnc_irls_p_rscale*sigma_pop*gnc_irls_p_epsilon_scale
            gnc_irls_p_epsilon_limit = 1.0
            gnc_irls_p_p = 0.0
            gnc_irls_p_beta = 0.8

            evaluator_instance = LinearRegressorGNC_IRLSpEvaluator(data[0])

            gnc_irls_p_instance = IRLS(GNC_IRLSpParams(GNC_IRLSpInfluenceFunc(),
                                                       gnc_irls_p_p, gnc_irls_p_rscale, gnc_irls_p_epsilon_base,
                                                       epsilon_limit=gnc_irls_p_epsilon_limit, beta=gnc_irls_p_beta),
                                       data, evaluator_instance=evaluator_instance, weight=weight)
            start_time = time.time()
            gnc_irls_p_instance.run() # this can fail but let's use the result anyway
            alg_result.time_gnc_irls_p += time.time()-start_time
            m_gnc_irls_p = gnc_irls_p_instance.final_model

            vec_gnc_irls_p += math.pow(m_gnc_irls_p-alg_result.m_gt, 2.0)
            gnc_irlsp_supgn_instance = SupGaussNewton(GNC_IRLSpParams(GNC_IRLSpInfluenceFunc(),
                                                                      gnc_irls_p_p, gnc_irls_p_rscale, gnc_irls_p_epsilon_base,
                                                                      epsilon_limit=gnc_irls_p_epsilon_limit, beta=gnc_irls_p_beta),
                                                      data, evaluator_instance=evaluator_instance, weight=weight)

            start_time = time.time()
            m_rme = M_estimator(data, beta=rme_beta_scale*sigma_pop)
            alg_result.time_rme += time.time()-start_time
            vec_rme += math.pow(m_rme-alg_result.m_gt, 2.0)

        if output_file_1 is not None:
            # get min and max of data
            y_min = y_max = 0.0
            x_min = x_max = None
            # override x limit 
            #x_min = 0.7
            #x_max = 1.3

            if x_min is None:
                dmin = min(data)
                dmax = max(data)

                # allow border
                drange = dmax-dmin
                x_min = dmin - 0.05*drange
                x_max = dmax + 0.05*drange

            mlist = np.linspace(x_min, x_max, num=300)
            for mx in mlist:
                y_max = max(y_max, objective_func(mx, welsch_supgn_instance))

            y_min *= 1.1 # allow for a small border
            y_max *= 1.1 # allow for a small border            

            plt.close("all")
            plt.figure(num=1, dpi=240)
            ax = plt.gca()
            #plt.box(False)
            ax.set_ylim((y_min, y_max))

            rmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
            if smoothie:
                gncs_draw_curve(plt, rmfv(mlist, optimiser_instance=welsch_supgn_instance), ("SupGN", "Welsch", "GNC_Welsch"), xvalues=mlist, draw_markers=False, hlight_x_value=m_gnc_welsch, ax=ax)
            else:
                gncs_draw_curve(plt, rmfv(mlist, optimiser_instance=welsch_irls_instance), ("IRLS", "Welsch", "GNC_Welsch"), xvalues=mlist, draw_markers=False, hlight_x_value=m_gnc_welsch, ax=ax)

            gncs_draw_vline(plt, alg_result.m_gt, ("GroundTruth", "", ""))
            if smoothie:
                gncs_draw_vline(plt, m_gnc_welsch, ("SupGN", "Welsch", "GNC_Welsch"), use_label=False)
            else:
                gncs_draw_vline(plt, m_gnc_welsch, ("IRLS", "Welsch", "GNC_Welsch"), use_label=False)

            if show_others:
                if not smoothie:
                    gncs_draw_vline(plt, mean,           ("Mean",   "Basic",       ""          ))

                gncs_draw_vline(plt, m_trimmed,      ("Mean",   "Trimmed",     ""          ))
                gncs_draw_vline(plt, median,         ("Median", "Basic",       ""          ))
                gncs_draw_vline(plt, trimean,        ("Trimean", "Basic",       ""          ))

            gncs_draw_data_points(plt, data, x_min, x_max, n0, weight=weight)

            plt.legend()
            plt.savefig(output_file_1, bbox_inches='tight')
            plt.show()            

            if output_file_2 is not None:
                plt.close("all")
                plt.figure(num=1, dpi=240)
                ax = plt.gca()
                #plt.box(False)
                ax.set_ylim((y_min, y_max))

                rmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
                gncs_draw_curve(plt, rmfv(mlist, optimiser_instance=welsch_irls_instance), ("IRLS", "Welsch", "GNC_Welsch"), xvalues=mlist, draw_markers=False, hlight_x_value=m_gnc_welsch, ax=ax)

                hmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
                hmfv_scaled = hmfv(mlist, optimiser_instance=pseudo_huber_supgn_instance)
                hmfv_scaled *= 0.5
                gncs_draw_curve(plt, hmfv_scaled, ("IRLS", "PseudoHuber", "Welsch"), xvalues=mlist, draw_markers=False, hlight_x_value=m_pseudo_huber, ax=ax)

                gmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
                gmfv_scaled = gmfv(mlist, optimiser_instance=gnc_irlsp_supgn_instance)
                gmfv_scaled *= 0.1
                gncs_draw_curve(plt, gmfv_scaled, ("IRLS", "GNC_IRLSp", "GNC_IRLSp0"), xvalues=mlist, draw_markers=False, hlight_x_value=m_gnc_irls_p, ax=ax)

                gncs_draw_vline(plt, alg_result.m_gt, ("GroundTruth", "", ""))
                if smoothie:
                    gncs_draw_vline(plt, m_gnc_welsch, ("SupGN", "Welsch", "GNC_Welsch"), use_label=False)
                else:
                    gncs_draw_vline(plt, m_gnc_welsch, ("IRLS", "Welsch", "GNC_Welsch"), use_label=False)

                if show_others and not smoothie:
                    gncs_draw_vline(plt, m_pseudo_huber, ("IRLS",   "PseudoHuber", "Welsch"    ), use_label=False)
                    gncs_draw_vline(plt, m_gnc_irls_p,   ("IRLS",   "GNC_IRLSp",   "GNC_IRLSp0"), use_label=False)
                    gncs_draw_vline(plt, m_rme,          ("RME",    "",            ""          ))

                gncs_draw_data_points(plt, data, x_min, x_max, n0, weight=weight)

                plt.legend()
                plt.savefig(output_file_1, bbox_inches='tight')
                plt.show()  

    norm = 1.0/(alg_result.n_samples-1)
    alg_result.sd_gnc_welsch = math.sqrt(vec_gnc_welsch*norm)
    alg_result.sd_trimmed = math.sqrt(vec_trimmed*norm)
    alg_result.sd_median = math.sqrt(vec_median*norm)
    alg_result.sd_trimean = math.sqrt(vec_trimean*norm)
    if not smoothie:
        alg_result.sd_mean = math.sqrt(vec_mean*norm)
        alg_result.sd_huber = math.sqrt(vec_pseudo_huber*norm)
        alg_result.sd_gnc_irls_p = math.sqrt(vec_gnc_irls_p*norm)
        alg_result.sd_rme = math.sqrt(vec_rme*norm)

    return alg_result
