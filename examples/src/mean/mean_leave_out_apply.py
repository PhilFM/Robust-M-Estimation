import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from pathlib import Path

from robust_mean import M_estimator

from gnc_smoothie_philfm.irls import IRLS
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.gnc_null_params import GNC_NullParams
from gnc_smoothie_philfm.gnc_irls_p_params import GNC_IRLSpParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie_philfm.pseudo_huber_influence_func import PseudoHuberInfluenceFunc
from gnc_smoothie_philfm.gnc_irls_p_influence_func import GNC_IRLSpInfluenceFunc
from gnc_smoothie_philfm.draw_functions import gncs_draw_data_points
from gnc_smoothie_philfm.plt_alg_vis import gncs_draw_vline, gncs_draw_curve, gncs_draw_histogram
from gnc_smoothie_philfm.cython_files.linear_regressor_welsch_evaluator import LinearRegressorWelschEvaluator
from gnc_smoothie_philfm.cython_files.linear_regressor_pseudo_huber_evaluator import LinearRegressorPseudoHuberEvaluator
from gnc_smoothie_philfm.cython_files.linear_regressor_gnc_irls_p_evaluator import LinearRegressorGNC_IRLSpEvaluator

from trimmed_mean import trimmed_mean
from tukey_trimean import tukey_trimean

show_others = True

def objective_func(m, optimiser_instance) -> float:
    if type(m) is not np.ndarray:
        m = [m]

    return optimiser_instance.objective_func(np.array(m))

def update_limits(
        vlist: list[float],
        dmin: float,
        dmax: float
) -> tuple[float,float]: 
    if vlist is not None:
        if dmin is None:
            dmin = min(vlist)
            dmax = max(vlist)
        else:
            dmin = min(dmin, min(vlist))
            dmax = max(dmax, max(vlist))

    return dmin,dmax

class StatsResult:
    m_gnc_welsch: float = None
    mean: float = None
    m_pseudo_huber: float = None
    m_trimmed: float = None
    median: float = None
    trimean: float = None
    m_gnc_irls_p: float=None
    m_rme: float=None

    # instances for plotting other results
    welsch_instance = None
    pseudo_huber_instance = None
    gnc_irls_p_instance = None

def calculate_stats(
        data: np.ndarray,
        sigma_pop: float,
        welsch_q: float,
        pseudo_huber_sigma_scale: float,
        gnc_irls_p_epsilon_scale: float,
        rme_beta_scale: float,
        test_run: bool
) -> StatsResult:
    welsch_sigma = sigma_pop/welsch_q
    evaluator_instance = LinearRegressorWelschEvaluator(data[0])

    x_min = min(data)
    x_max = max(data)
    drange = x_max - x_min
    sigma_limit = drange

    # for the mean estimation paper let's use IRLS
    stats_result = StatsResult()
    stats_result.welsch_instance = IRLS(GNC_WelschParams(WelschInfluenceFunc(),
                                                         welsch_sigma,
                                                         sigma_limit, 50),
                                        data,
                                        evaluator_instance=evaluator_instance,
                                        max_niterations=5000,
                                        diff_thres=1.e-12*sigma_pop)
    #                                    messages_file=sys.stdout)
    if stats_result.welsch_instance.run():
        stats_result.m_gnc_welsch = stats_result.welsch_instance.final_model[0]
        if False: #not test_run:
            mlist = np.linspace(x_min, x_max, num=300)
            y_min = 0.0
            y_max = max(objective_func(mx, stats_result.welsch_instance) for mx in mlist)

            y_min *= 1.1 # allow for a small border
            y_max *= 1.1 # allow for a small border            

            plt.close("all")
            plt.figure(num=1, dpi=240)
            ax = plt.gca()
            #plt.box(False)
            ax.set_ylim((y_min, y_max))

            rmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
            gncs_draw_curve(plt, rmfv(mlist, optimiser_instance=stats_result.welsch_instance), ("SupGN", "Welsch", "GNC_Welsch"),
                            xvalues=mlist, draw_markers=False, hlight_x_value=stats_result.m_gnc_welsch, ax=ax)

            gncs_draw_data_points(plt, data, x_min, x_max, len(data))

            plt.legend()
            plt.show()            

    stats_result.mean = evaluator_instance.weighted_fit([data], [np.ones(len(data))])[0][0]

    evaluator_instance = LinearRegressorPseudoHuberEvaluator(data[0])
    pseudo_huber_sigma = pseudo_huber_sigma_scale*sigma_pop
    stats_result.pseudo_huber_instance = IRLS(GNC_NullParams(PseudoHuberInfluenceFunc(sigma=pseudo_huber_sigma)),
                                              data,
                                              evaluator_instance=evaluator_instance,
                                              diff_thres=1.e-12*sigma_pop)
    stats_result.pseudo_huber_instance.run()  # this can fail but let's use the result anyway
    stats_result.m_pseudo_huber = stats_result.pseudo_huber_instance.final_model[0]

    trim_size = len(data)//4
    stats_result.m_trimmed = trimmed_mean(data, trim_size)

    stats_result.median = np.median(data)

    stats_result.trimean = tukey_trimean(data)

    gnc_irls_p_rscale = 1.0/drange
    gnc_irls_p_epsilon_base = gnc_irls_p_rscale*sigma_pop*gnc_irls_p_epsilon_scale
    gnc_irls_p_epsilon_limit = 1.0
    gnc_irls_p_p = 0.0
    gnc_irls_p_beta = 0.8

    evaluator_instance = LinearRegressorGNC_IRLSpEvaluator(data[0])
    stats_result.gnc_irls_p_instance = IRLS(GNC_IRLSpParams(GNC_IRLSpInfluenceFunc(),
                                                            gnc_irls_p_p,
                                                            gnc_irls_p_rscale,
                                                            gnc_irls_p_epsilon_base,
                                                            gnc_irls_p_epsilon_limit,
                                                            gnc_irls_p_beta),
                                            data,
                                            evaluator_instance=evaluator_instance,
                                            diff_thres=1.e-12*sigma_pop)
    stats_result.gnc_irls_p_instance.run() # this can fail but let's use the result anyway
    stats_result.m_gnc_irls_p = stats_result.gnc_irls_p_instance.final_model[0]
    stats_result.m_rme = M_estimator(data, beta=rme_beta_scale*sigma_pop)
    return stats_result

class LeaveOutMeanAlgResult:
    # standard deviations
    sd_gnc_welsch: float = None
    sd_mean: float = None
    sd_huber: float = None
    sd_trimmed: float = None
    sd_median: float = None
    sd_trimean: float = None
    sd_gnc_irls_p: float = None
    sd_rme: float = None

    n0: int = None

def mean_leave_out_apply(data_full: np.array,
                         sigma_pop: float,
                         leave_out_fraction: float,
                         n_samples: int,
                         welsch_q: float,
                         pseudo_huber_sigma_scale: float,
                         gnc_irls_p_epsilon_scale: float,
                         rme_beta_scale: float,
                         stats_result_ref: StatsResult,
                         output_file_1: Path=None,
                         output_file_2: Path=None,
                         test_run:bool=False,
                         smoothie:bool=False
) -> LeaveOutMeanAlgResult:
    sum_gnc_welsch = sum_mean = sum_pseudo_huber = sum_trimmed = sum_median = sum_trimean = sum_gnc_irls_p = sum_rme = 0.0
    vec_gnc_welsch = vec_mean = vec_pseudo_huber = vec_trimmed = vec_median = vec_trimean = vec_gnc_irls_p = vec_rme = 0.0
    list_gnc_welsch = []
    list_mean = []
    list_pseudo_huber = []
    list_trimmed = []
    list_median = []
    list_trimean = []
    list_gnc_irls_p = []
    list_rme = []

    alg_result = LeaveOutMeanAlgResult()
    alg_result.n0 = int((1.0-leave_out_fraction)*len(data_full)+0.5)
    if not test_run:
        print("")
        print("welsch_q=",welsch_q,"pseudo_huber_sigma_scale=",pseudo_huber_sigma_scale,"gnc_irls_p_epsilon_scale=",gnc_irls_p_epsilon_scale,"rme_beta_scale=",rme_beta_scale)
        print("leave_out_fraction=",leave_out_fraction,"n0=",alg_result.n0)

    for i in range(n_samples):
        # randomly leave out data
        idx = []
        data = []
        for j in range(alg_result.n0):
            while True:
                idx_try = np.random.randint(len(data_full))
                found = False
                for k in idx:
                    if k == idx_try:
                        found = True
                        break

                if not found:
                    break

            idx.append(idx_try)
            data.append(data_full[idx_try])

        data = np.array(data)
        #print("data=",data)

        stats_result = calculate_stats(data, sigma_pop, welsch_q,
                                       pseudo_huber_sigma_scale, gnc_irls_p_epsilon_scale,
                                       rme_beta_scale, test_run if i < 2 else True)
        list_gnc_welsch.append(stats_result.m_gnc_welsch)
        sum_gnc_welsch += stats_result.m_gnc_welsch
        vec_gnc_welsch += stats_result.m_gnc_welsch ** 2

        list_mean.append(stats_result.mean)
        sum_mean += stats_result.mean
        vec_mean += stats_result.mean ** 2

        list_pseudo_huber.append(stats_result.m_pseudo_huber)
        sum_pseudo_huber += stats_result.m_pseudo_huber
        vec_pseudo_huber += stats_result.m_pseudo_huber ** 2

        list_trimmed.append(stats_result.m_trimmed[0])
        sum_trimmed += stats_result.m_trimmed[0]
        vec_trimmed += stats_result.m_trimmed[0] ** 2

        list_median.append(stats_result.median)
        sum_median += stats_result.median
        vec_median += stats_result.median ** 2

        list_trimean.append(stats_result.trimean)
        sum_trimean += stats_result.trimean
        vec_trimean += stats_result.trimean ** 2

        list_gnc_irls_p.append(stats_result.m_gnc_irls_p)
        sum_gnc_irls_p += stats_result.m_gnc_irls_p
        vec_gnc_irls_p += stats_result.m_gnc_irls_p ** 2

        list_rme.append(stats_result.m_rme[0])
        sum_rme += stats_result.m_rme[0]
        vec_rme += stats_result.m_rme[0] ** 2

    m_gt = stats_result_ref.m_gnc_welsch
    vec_gnc_welsch   += m_gt*(m_gt*len(list_gnc_welsch)   - 2.0*sum_gnc_welsch)

    m_gt = stats_result_ref.mean
    vec_mean         += m_gt*(m_gt*len(list_mean)         - 2.0*sum_mean)

    m_gt = stats_result_ref.m_pseudo_huber
    vec_pseudo_huber += m_gt*(m_gt*len(list_pseudo_huber) - 2.0*sum_pseudo_huber)

    m_gt = stats_result_ref.m_trimmed
    vec_trimmed      += m_gt*(m_gt*len(list_trimmed)      - 2.0*sum_trimmed)

    m_gt = stats_result_ref.median
    vec_median       += m_gt*(m_gt*len(list_median)       - 2.0*sum_median)

    m_gt = stats_result_ref.trimean
    vec_trimean       += m_gt*(m_gt*len(list_trimean)       - 2.0*sum_trimean)

    m_gt = stats_result_ref.m_gnc_irls_p
    vec_gnc_irls_p   += m_gt*(m_gt*len(list_gnc_irls_p)   - 2.0*sum_gnc_irls_p)

    m_gt = stats_result_ref.m_rme
    vec_rme          += m_gt*(m_gt*len(list_rme)          - 2.0*sum_rme)

    if False: #not test_run:
        dmin = dmax = None
        dmin,dmax = update_limits(list_gnc_welsch,   dmin, dmax)
        dmin,dmax = update_limits(list_mean,         dmin, dmax)
        dmin,dmax = update_limits(list_pseudo_huber, dmin, dmax)
        dmin,dmax = update_limits(list_trimmed,      dmin, dmax)
        dmin,dmax = update_limits(list_median,       dmin, dmax)
        dmin,dmax = update_limits(list_trimean,      dmin, dmax)
        dmin,dmax = update_limits(list_gnc_irls_p,   dmin, dmax)
        dmin,dmax = update_limits(list_rme,          dmin, dmax)
        drange = dmax-dmin
        x_min = dmin - 0.1*drange
        x_max = dmax + 0.1*drange

        plt.close("all")
        plt.figure(num=1, dpi=240)

        bin_size = 0.001*sigma_pop/(1.0-leave_out_fraction)
        if smoothie:
            gncs_draw_histogram(plt, x_min, x_max, bin_size, list_gnc_welsch, ("SupGN", "Welsch", "GNC_Welsch"))
        else:
            gncs_draw_histogram(plt, x_min, x_max, bin_size, list_gnc_welsch, ("IRLS", "Welsch", "GNC_Welsch"))

        if not smoothie:
            gncs_draw_histogram(plt, x_min, x_max, bin_size, list_mean, ("Mean", "Basic", ""))

        gncs_draw_histogram(plt, x_min, x_max, bin_size, list_trimmed, ("Mean", "Trimmed", ""))
        gncs_draw_histogram(plt, x_min, x_max, bin_size, list_median, ("Median", "Basic", ""))
        gncs_draw_histogram(plt, x_min, x_max, bin_size, list_trimean, ("Trimean", "Basic", ""))

        plt.legend()
        plt.show()

        plt.close("all")
        plt.figure(num=1, dpi=240)

        bin_size = 0.001*sigma_pop/(1.0-leave_out_fraction)
        if smoothie:
            gncs_draw_histogram(plt, x_min, x_max, bin_size, list_gnc_welsch, ("SupGN", "Welsch", "GNC_Welsch"))
        else:
            gncs_draw_histogram(plt, x_min, x_max, bin_size, list_gnc_welsch, ("IRLS", "Welsch", "GNC_Welsch"))

        gncs_draw_histogram(plt, x_min, x_max, bin_size, list_pseudo_huber, ("IRLS", "PseudoHuber", "Welsch"))
        gncs_draw_histogram(plt, x_min, x_max, bin_size, list_gnc_irls_p, ("IRLS", "GNC_IRLSp", "GNC_IRLSp0"))
        gncs_draw_histogram(plt, x_min, x_max, bin_size, list_rme, ("RME", "", ""))

        plt.legend()
        plt.show()
        
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
            y_max = max(objective_func(mx, welsch_instance) for mx in mlist)

        y_min *= 1.1 # allow for a small border
        y_max *= 1.1 # allow for a small border            

        plt.close("all")
        plt.figure(num=1, dpi=240)
        ax = plt.gca()
        #plt.box(False)
        #print("y_max=",y_max)
        ax.set_ylim((y_min, y_max))

        rmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
        gncs_draw_curve(plt, rmfv(mlist, optimiser_instance=welsch_instance), ("IRLS", "Welsch", "GNC_Welsch"), xvalues=mlist, draw_markers=False, hlight_x_value=m_gnc_welsch, ax=ax)

        gncs_draw_vline(plt, m_gt, ("GroundTruth", "", ""))
        if smoothie:
            gncs_draw_vline(plt, m_gnc_welsch, ("SupGN", "Welsch", "GNC_Welsch"), use_label=False)
        else:
            gncs_draw_vline(plt, m_gnc_welsch, ("IRLS", "Welsch", "GNC_Welsch"), use_label=False)

        if show_others:
            if not smoothie:
                gncs_draw_vline(plt, mean,           ("Mean",   "Basic",       ""          ))
                
            gncs_draw_vline(plt, m_trimmed,      ("Mean",   "Trimmed",     ""          ))
            gncs_draw_vline(plt, median,         ("Median", "Basic",       ""          ))
            gncs_draw_vline(plt, trimean,         ("Trimean", "Basic",       ""          ))

        gncs_draw_data_points(plt, data, x_min, x_max, alg_result.n0)

        plt.legend()
        plt.savefig(output_file_1, bbox_inches='tight')
        plt.show()            

        if output_file_2 is not None:
            plt.close("all")
            plt.figure(num=1, dpi=240)
            ax = plt.gca()
            #plt.box(False)
            #print("y_max=",y_max)
            ax.set_ylim((y_min, y_max))

            rmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
            gncs_draw_curve(plt, rmfv(mlist, optimiser_instance=welsch_instance), ("IRLS", "Welsch", "GNC_Welsch"), xvalues=mlist, draw_markers=False, hlight_x_value=m_gnc_welsch, ax=ax)

            hmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
            hmfv_scaled = hmfv(mlist, optimiser_instance=pseudo_huber_instance)
            hmfv_scaled *= 0.5
            gncs_draw_curve(plt, hmfv_scaled, ("IRLS", "PseudoHuber", "Welsch"), xvalues=mlist, draw_markers=False, hlight_x_value=m_pseudo_huber, ax=ax)

            gmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
            gmfv_scaled = gmfv(mlist, optimiser_instance=gnc_irls_p_instance)
            gmfv_scaled *= 0.1
            gncs_draw_curve(plt, gmfv_scaled, ("IRLS", "GNC_IRLSp", "GNC_IRLSp0"), xvalues=mlist, draw_markers=False, hlight_x_value=m_gnc_irls_p, ax=ax)

            gncs_draw_vline(plt, m_gt, ("GroundTruth", "", ""))
            if smoothie:
                gncs_draw_vline(plt, m_gnc_welsch, ("SupGN", "Welsch", "GNC_Welsch"), use_label=False)
            else:
                gncs_draw_vline(plt, m_gnc_welsch, ("IRLS", "Welsch", "GNC_Welsch"), use_label=False)

            if show_others:
                if not smoothie:
                    gncs_draw_vline(plt, m_pseudo_huber, ("IRLS",   "PseudoHuber", "Welsch"    ), use_label=False)
                    gncs_draw_vline(plt, m_gnc_irls_p,   ("IRLS",   "GNC_IRLSp",   "GNC_IRLSp0"), use_label=False)
                    gncs_draw_vline(plt, m_rme,          ("RME",    "",            ""          ))

            gncs_draw_data_points(plt, data, x_min, x_max, alg_result.n0)

            plt.legend()
            plt.savefig(output_file_2, bbox_inches='tight')
            plt.show()            

    # estimate sample variance from Sx and Sxx
    # Estimated mean is m = Sx/n_samples
    # Standard deviation estimate is then sd = sqrt(sum((x-m)^2)/(n_samples-1))
    #                                        = sqrt((Sxx - 2*m*Sx + n_samples*m^2)/(n_samples-1))
    #                                        = sqrt((Sxx - Sx^2/n_samples)/(n_samples-1))
    norm = 1.0/(n_samples-1)
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
