import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

from robust_mean import M_estimator

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
from gnc_smoothie.cython_files.linear_regressor_welsch_evaluator import LinearRegressorWelschEvaluator
from gnc_smoothie.cython_files.linear_regressor_pseudo_huber_evaluator import LinearRegressorPseudoHuberEvaluator
from gnc_smoothie.cython_files.linear_regressor_gnc_irls_p_evaluator import LinearRegressorGNC_IRLSpEvaluator

from trimmed_mean import trimmed_mean
from tukey_trimean import tukey_trimean
from save_sample import save_sample

show_others = True

def objective_func(m, optimiser_instance):
    return optimiser_instance.objective_func([m])

class CompareMeanAlgResult:
    def __init__(
        self
    ):
        # variances
        self.var_gnc_welsch: float = 0.0
        self.var_mean: float = 0.0
        self.var_huber: float = 0.0
        self.var_trimmed: float = 0.0
        self.var_median: float = 0.0
        self.var_trimean: float = 0.0
        self.var_gnc_irls_p: float = 0.0
        self.var_rme: float = 0.0

        # timing
        self.av_time_gnc_welsch: float = 0.0
        self.av_time_mean: float = 0.0
        self.av_time_huber: float = 0.0
        self.av_time_trimmed: float = 0.0
        self.av_time_median: float = 0.0
        self.av_time_trimean: float = 0.0
        self.av_time_gnc_irls_p: float = 0.0
        self.av_time_rme: float = 0.0
    
        self.n_samples = None

def mean_compare_apply(sigma_pop: float,
                       xgtrange: float,
                       n: int,
                       n_samples_base: int,
                       min_n_samples: int,
                       outlier_fraction: float,
                       welsch_q: float,
                       huber_sigma_scale: float,
                       gnc_irls_p_epsilon_scale: float,
                       rme_beta_scale: float,
                       student_t_dof: int = 0,
                       output_file_1:Path = None,
                       output_file_2:Path = None,
                       test_run:bool=False,
                       output_folder:str = None,
                       use_supgn:bool = True,
                       just_simple_algorithms:bool=False) -> CompareMeanAlgResult:
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

        if False: #sample < 10:
            save_sample(data, sigma_pop, output_folder,
                        "compare_sample_n" + str(n) + "_range" + str(int(xgtrange)) + "_of" + str(int(100.0*outlier_fraction)) + "_sd" + str(student_t_dof) + "_" + str(sample+1) + ".png")

        # Sup-GN or GNC IRLS-Welsch
        evaluator_instance = LinearRegressorWelschEvaluator(data[0])

        gnc_welsch_sigma = sigma_pop/welsch_q
        if use_supgn:
            # Sup-GN
            welsch_supgn_instance = SupGaussNewton(
                GNC_WelschParams(
                    WelschInfluenceFunc(),
                    gnc_welsch_sigma,
                    sigma_limit=max(max(data)-min(data),xgtrange,10.0*sigma_pop),
                    num_sigma_steps=10),
                evaluator_instance=evaluator_instance,
                max_niterations=200)
            start_time = time.time()
            welsch_supgn_instance.fit(data, weight=weight) # may fail
            m_gnc_welsch = welsch_supgn_instance.final_model[0]
            alg_result.av_time_gnc_welsch += time.time()-start_time
        else:
            # use IRLS
            welsch_irls_instance = IRLS(
                GNC_WelschParams(
                    WelschInfluenceFunc(),
                    gnc_welsch_sigma,
                    sigma_limit=max(max(data)-min(data),xgtrange,10.0*sigma_pop),
                    num_sigma_steps=10),
                evaluator_instance=evaluator_instance,
                max_niterations=200)
            start_time = time.time()
            welsch_irls_instance.fit(data, weight=weight,) # may fail
            m_gnc_welsch = welsch_irls_instance.final_model[0]
            alg_result.av_time_gnc_welsch += time.time()-start_time

        alg_result.var_gnc_welsch += (m_gnc_welsch-alg_result.m_gt) ** 2.0

        # trimmed mean
        if student_t_dof == 0:
            # correct trim_size to match level of outliers would be 0.5*outlier_fraction*n, but this assumes that the outliers
            # are evenly distributed above and below the good data. To allow for all the outliers to be below (or above) the
            # good data would require a trim size of outlier_fraction*n, but this is overly pessimistic.
            # So let's compromise with 0.75.
            trim_size = int(0.5 + 0.75*outlier_fraction*n)
        else:
            trim_size = n//4

        start_time = time.time()
        tmean = trimmed_mean(data, trim_size=trim_size, weight=weight)
        m_trimmed = tmean[0]
        alg_result.av_time_trimmed += time.time()-start_time
        alg_result.var_trimmed += (m_trimmed-alg_result.m_gt) ** 2.0

        # median
        start_time = time.time()
        median = np.median(data)
        alg_result.av_time_median += time.time()-start_time
        alg_result.var_median += (median-alg_result.m_gt) ** 2.0

        # trimean
        start_time = time.time()
        trimean = tukey_trimean(data)
        alg_result.av_time_trimean += time.time()-start_time
        alg_result.var_trimean += (trimean-alg_result.m_gt) ** 2.0

        if not just_simple_algorithms:
            # arithmetic mean
            mean = evaluator_instance.weighted_fit([data], [weight])
            alg_result.var_mean += (mean[0]-alg_result.m_gt) ** 2.0

            evaluator_instance = LinearRegressorPseudoHuberEvaluator(data[0])

            # Pseudo-Huber
            huber_sigma = sigma_pop*huber_sigma_scale
            if use_supgn:
                huber_instance = SupGaussNewton(GNC_NullParams(PseudoHuberInfluenceFunc(sigma=huber_sigma)),
                                                evaluator_instance=evaluator_instance)
            else:
                huber_instance = IRLS(GNC_NullParams(PseudoHuberInfluenceFunc(sigma=huber_sigma)),
                                      evaluator_instance=evaluator_instance)

            start_time = time.time()
            huber_instance.fit(data, weight=weight)  # this can fail but let's use the result anyway
            alg_result.av_time_huber += time.time()-start_time
            m_huber = huber_instance.final_model

            alg_result.var_huber += (m_huber-alg_result.m_gt) ** 2.0

            # GNC IRLS-p
            gnc_irls_p_rscale = 1.0/xgtrange
            gnc_irls_p_epsilon_base = gnc_irls_p_rscale*sigma_pop*gnc_irls_p_epsilon_scale
            gnc_irls_p_epsilon_limit = 1.0
            gnc_irls_p_p = 0.0
            gnc_irls_p_beta = 0.8

            evaluator_instance = LinearRegressorGNC_IRLSpEvaluator(data[0])

            if use_supgn:
                gnc_irls_p_instance = SupGaussNewton(GNC_IRLSpParams(GNC_IRLSpInfluenceFunc(),
                                                                     gnc_irls_p_p, gnc_irls_p_rscale, gnc_irls_p_epsilon_base,
                                                                     epsilon_limit=gnc_irls_p_epsilon_limit, beta=gnc_irls_p_beta),
                                                     evaluator_instance=evaluator_instance)
            else:
                gnc_irls_p_instance = IRLS(GNC_IRLSpParams(GNC_IRLSpInfluenceFunc(),
                                                           gnc_irls_p_p, gnc_irls_p_rscale, gnc_irls_p_epsilon_base,
                                                           epsilon_limit=gnc_irls_p_epsilon_limit, beta=gnc_irls_p_beta),
                                           evaluator_instance=evaluator_instance)
            start_time = time.time()
            gnc_irls_p_instance.fit(data, weight=weight) # this can fail but let's use the result anyway
            alg_result.av_time_gnc_irls_p += time.time()-start_time
            m_gnc_irls_p = gnc_irls_p_instance.final_model

            alg_result.var_gnc_irls_p += (m_gnc_irls_p-alg_result.m_gt) ** 2.0
 
            # RME
            start_time = time.time()
            m_rme = M_estimator(data, beta=rme_beta_scale*sigma_pop)
            alg_result.av_time_rme += time.time()-start_time
            alg_result.var_rme += (m_rme-alg_result.m_gt) ** 2.0

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
            if use_supgn:
                gncs_draw_curve(plt, rmfv(mlist, optimiser_instance=welsch_supgn_instance), ("SupGN", "Welsch", "GNC_Welsch"), xvalues=mlist, draw_markers=False, hlight_x_value=m_gnc_welsch, ax=ax)
            else:
                gncs_draw_curve(plt, rmfv(mlist, optimiser_instance=welsch_irls_instance), ("IRLS", "Welsch", "GNC_Welsch"), xvalues=mlist, draw_markers=False, hlight_x_value=m_gnc_welsch, ax=ax)

            gncs_draw_vline(plt, alg_result.m_gt, ("GroundTruth", "", ""))
            if use_supgn:
                gncs_draw_vline(plt, m_gnc_welsch, ("SupGN", "Welsch", "GNC_Welsch"), use_label=False)
            else:
                gncs_draw_vline(plt, m_gnc_welsch, ("IRLS", "Welsch", "GNC_Welsch"), use_label=False)

            if show_others:
                if not just_simple_algorithms:
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

                alg_id = "SupGN" if use_supgn else "IRLS"

                rmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
                gncs_draw_curve(plt, rmfv(mlist, optimiser_instance=welsch_irls_instance), (alg_id, "Welsch", "GNC_Welsch"), xvalues=mlist, draw_markers=False, hlight_x_value=m_gnc_welsch, ax=ax)

                hmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
                hmfv_scaled = hmfv(mlist, optimiser_instance=huber_instance)
                hmfv_scaled *= 0.5
                gncs_draw_curve(plt, hmfv_scaled, (alg_id, "PseudoHuber", "PseudoHuber"), xvalues=mlist, draw_markers=False, hlight_x_value=m_huber, ax=ax)

                gmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
                gmfv_scaled = gmfv(mlist, optimiser_instance=gnc_irls_p_instance)
                gmfv_scaled *= 0.1
                gncs_draw_curve(plt, gmfv_scaled, (alg_id, "GNC_IRLSp", "GNC_IRLSp0"), xvalues=mlist, draw_markers=False, hlight_x_value=m_gnc_irls_p, ax=ax)

                gncs_draw_vline(plt, alg_result.m_gt, ("GroundTruth", "", ""))
                gncs_draw_vline(plt, m_gnc_welsch, (alg_id, "Welsch", "GNC_Welsch"), use_label=False)

                if show_others and not just_simple_algorithms:
                    gncs_draw_vline(plt, m_huber,        (alg_id,   "PseudoHuber", "Welsch"    ), use_label=False)
                    gncs_draw_vline(plt, m_gnc_irls_p,   (alg_id,   "GNC_IRLSp",   "GNC_IRLSp0"), use_label=False)
                    gncs_draw_vline(plt, m_rme,          ("RME",    "",            ""          ))

                gncs_draw_data_points(plt, data, x_min, x_max, n0, weight=weight)

                plt.legend()
                plt.savefig(output_file_1, bbox_inches='tight')
                plt.show()  

    norm = 1.0/(alg_result.n_samples-1)
    alg_result.var_gnc_welsch *= norm
    alg_result.var_trimmed *= norm
    alg_result.var_median *= norm
    alg_result.var_trimean *= norm
    if not just_simple_algorithms:
        alg_result.var_mean *= norm
        alg_result.var_huber *= norm
        alg_result.var_gnc_irls_p *= norm
        alg_result.var_rme *= norm

    alg_result.av_time_gnc_welsch /= alg_result.n_samples
    alg_result.av_time_trimmed /= alg_result.n_samples
    alg_result.av_time_median /= alg_result.n_samples
    alg_result.av_time_trimean /= alg_result.n_samples
    if not just_simple_algorithms:
        alg_result.av_time_mean /= alg_result.n_samples
        alg_result.av_time_huber /= alg_result.n_samples
        alg_result.av_time_gnc_irls_p /= alg_result.n_samples
        alg_result.av_time_rme /= alg_result.n_samples
    
    return alg_result
