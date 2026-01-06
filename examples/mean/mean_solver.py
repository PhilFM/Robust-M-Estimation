import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.irls import IRLS
from gnc_smoothie_philfm.gnc_null_params import GNC_NullParams
from gnc_smoothie_philfm.linear_model.linear_regressor import LinearRegressor

# Welsch
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie_philfm.linear_model.linear_regressor_welsch import LinearRegressorWelsch

# Pseudo-Huber
from gnc_smoothie_philfm.pseudo_huber_influence_func import PseudoHuberInfluenceFunc
from gnc_smoothie_philfm.linear_model.linear_regressor_pseudo_huber import LinearRegressorPseudoHuber

# Geman-McClure
from gnc_smoothie_philfm.geman_mcclure_influence_func import GemanMcClureInfluenceFunc

# GNC IRLS-p
from gnc_smoothie_philfm.gnc_irls_p_influence_func import GNC_IRLSpInfluenceFunc
from gnc_smoothie_philfm.gnc_irls_p_params import GNC_IRLSpParams
from gnc_smoothie_philfm.linear_model.linear_regressor_gnc_irls_p import LinearRegressorGNC_IRLSp

# number of intermediate GNC curves to draw
n_intermediate_gnc_curves = 10

def objective_func(m, optimiser_instance, offset:float=0.0):
    if offset == 0.0:
        return optimiser_instance.objective_func([m])
    else:
        return offset - optimiser_instance.objective_func([m])

def get_y_limits(mlist: list, optimiser_instance, offset: float=0.0) -> (float,float):
    # get min and max of data
    y_min = y_max = 0.0
    y_max = max(objective_func(mx, optimiser_instance, offset) for mx in mlist)
    y_min *= 1.05 # allow for a small border
    y_max *= 1.05 # allow for a small border
    return y_min,y_max

def plot_gnc_example(sup_gn_instance, mean_est: float, x_range: float, final_weight: np.array, offset: float,
                     influence_func_name: str, test_run: bool, output_folder: str, output_file_name: str) -> None:
    mlist = np.linspace(0.0, x_range, num=300)
    fid_sign = (offset - 0.5)*sup_gn_instance.objective_func_sign()
    sup_gn_instance._param_instance.reset(True if fid_sign > 0.0 else False)
    (y_min,y_max) = get_y_limits(mlist, sup_gn_instance, offset)

    plt.close("all")
    plt.figure(num=1, dpi=240)
    ax = plt.gca()
    #plt.box(False)
    ax.set_xlim((0.0, x_range))
    ax.set_ylim((y_min, y_max))

    hmfv = np.vectorize(objective_func, excluded={"optimiser_instance","offset"})

    # plot some GNC intermediate objective curves
    last_xy = None
    last_color = None
    sup_gn_instance._param_instance.reset(True) # set to limit value
    n_steps = sup_gn_instance._param_instance.n_steps()
    for step in range(n_steps):
        alpha = (n_steps-1-step)/(n_steps-1)
        color = (0.8*alpha, 1.0, 0.0)
        plt.plot(mlist, hmfv(mlist, optimiser_instance=sup_gn_instance, offset=offset), lw = 1.0, color=color)

        # mark the maximum
        idx = np.argmax([ objective_func(mx, sup_gn_instance) for mx in mlist])
        mx = mlist[idx]
        this_xy = (mx, objective_func(mx, sup_gn_instance, offset))
        plt.plot(this_xy[0], this_xy[1], "o", color=color, markersize=2)
        if last_xy is not None:
            plt.plot((last_xy[0],this_xy[0]), (last_xy[1],this_xy[1]), color=last_color, linestyle="dotted", linewidth=0.5)

        sup_gn_instance._param_instance.increment()
        last_xy = this_xy
        last_color = color

    plt.plot(mlist, hmfv(mlist, optimiser_instance=sup_gn_instance, offset=offset), lw = 1.0, color="green", label=influence_func_name + " objective function")

    # mark the maximum
    idx = np.argmax([ objective_func(mx, sup_gn_instance) for mx in mlist])
    mx = mlist[idx]
    this_xy = (mx, objective_func(mx, sup_gn_instance, offset))
    plt.plot(this_xy[0], this_xy[1], "o", color="green", markersize=2)
    plt.plot((last_xy[0],this_xy[0]), (last_xy[1],this_xy[1]), color=last_color, linestyle="dotted", linewidth=0.5)

    # add invisible line and marker just for legend
    alpha = 0.5
    color = (0.8*alpha, 1.0, 0.0)
    plt.axvline(x = -1000, color = color, ymax = 0.1, lw = 1.0, label="GNC intermediates")
    plt.plot(-1000,-1000, "o", color=color, markersize=2, label="Intermediate maxima" if offset == 0.0 else "Intermediate minima")

    # draw data points as short vertical lines
    data = sup_gn_instance._data[0]
    plt.axvline(x = data[0][0], color = (1,0,0), ymax = 0.1, lw = 1.0, label="Inlier data values") # will be overwritten with corrected colour
    plt.axvline(x = data[0][0], color = (0,0,1), ymax = 0.1, lw = 1.0, label="Outlier data values") # will be overwritten with corrected colour
    max_weight = max(final_weight)
    for d,w in zip(data,final_weight, strict=True):
        alpha = w/max_weight
        color = [alpha, 0.0, 1.0-alpha]
        plt.axvline(x = d, color = color, ymax = 0.1, lw = 1.0)

    plt.axvline(x = mean_est, color = 'limegreen', label = 'Estimated mean', lw = 1.0)

    plt.legend()
    plt.savefig(os.path.join(output_folder, output_file_name), bbox_inches='tight')
    if not test_run:
        plt.show()

def mean_welsch_solver(data: np.array, scale: np.array, x_range: float, sigma_pop: float,
                       test_run: bool, output_folder: str) -> None:
    # estimation parameters
    p = 0.66667 # ratio of population standard deviation to base sigma value in estimation
    sigma_base = sigma_pop/p
    sigma_limit = x_range
    num_sigma_steps = 20
    max_niterations = 50

    model_instance = LinearRegressor(data[0])
    param_instance = GNC_WelschParams(WelschInfluenceFunc(), sigma_base, sigma_limit=sigma_limit,
                                      num_sigma_steps=num_sigma_steps)
    irls_instance = IRLS(param_instance, data, model_instance=model_instance, max_niterations=max_niterations, print_warnings=False)
    if irls_instance.run():
        m = irls_instance.final_model[0]
        if not test_run:
            print("Welsch IRLS result: m=", m)

    mean_finder = LinearRegressorWelsch(sigma_base, sigma_limit=sigma_limit, num_sigma_steps=num_sigma_steps,
                                        max_niterations=max_niterations, print_warnings=False, debug=True)
    if mean_finder.run(data):
        m = mean_finder.final_intercept[0]
        final_weight = mean_finder.final_weight
        if not test_run:
            print("Welsch Sup-GN optimisation result: m=", m)
            print("  final weights:",final_weight)

    # check result when scale is included
    if mean_finder.run(data, scale=scale):
        mscale = mean_finder.final_intercept[0]
        if not test_run:
            print("Welsch scale result difference=", mscale-m)

    if output_folder is not None:
        # create GNC instance for plotting
        param_instance_plot = GNC_WelschParams(WelschInfluenceFunc(), sigma_base, sigma_limit=sigma_limit,
                                               num_sigma_steps=n_intermediate_gnc_curves)
        sup_gn_instance_plot = SupGaussNewton(param_instance_plot, data, model_instance=model_instance)

        plot_gnc_example(sup_gn_instance_plot, m, x_range, final_weight, 0.0, # RobustAverage paper
                         "Welsch", test_run, output_folder, "mean_welsch.png")
        plot_gnc_example(sup_gn_instance_plot, m, x_range, final_weight, len(data), # for Smoothie paper,
                         "Welsch", test_run, output_folder, "mean_welsch_smoothie.png")

def mean_pseudo_huber_solver(data: np.array, scale: np.array, x_range: float, sigma_pop: float,
                             test_run: bool, output_folder: str) -> None:
    model_instance = LinearRegressor(data[0])
    sigma=2.0*sigma_pop
    influence_func_instance = PseudoHuberInfluenceFunc(sigma)
    param_instance = GNC_NullParams(influence_func_instance)
    irls_instance = IRLS(param_instance, data, model_instance=model_instance, max_niterations=200, print_warnings=False, debug=True)
    if irls_instance.run():
        m = irls_instance.final_model[0]
        final_weight = irls_instance.final_weight
        if not test_run:
            print("Pseudo-Huber result: m=", m)
            print("  final_weight=",final_weight)

    # check IRLS with scale
    irls_instance = IRLS(GNC_NullParams(influence_func_instance), data, model_instance=model_instance, scale=scale)
    if irls_instance.run():
        mscale = irls_instance.final_model[0]
        if not test_run:
            print("Pseudo-Huber scale result difference=", mscale-m)

    mean_finder = LinearRegressorPseudoHuber(sigma, print_warnings=False, debug=True)
    if mean_finder.run(data):
        m = mean_finder.final_intercept[0]
        final_weight = mean_finder.final_weight
        if not test_run:
            print("Pseudo-Huber Linear Regression optimisation result: m=", m)
            print("  final weights:", final_weight)

    if output_folder is not None:
        # for graph plotting
        sup_gn_instance = SupGaussNewton(param_instance, data, model_instance=model_instance, print_warnings=False)
        mlist = np.linspace(0.0, x_range, num=300)
        (y_min,y_max) = get_y_limits(mlist, sup_gn_instance)

        plt.close("all")
        plt.figure(num=1, dpi=240)
        ax = plt.gca()
        #plt.box(False)
        ax.set_xlim((0.0, x_range))
        ax.set_ylim((y_min, y_max))

        hmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})
        plt.plot(mlist, hmfv(mlist, optimiser_instance=sup_gn_instance), lw = 1.0, color="g", label="Pseudo-Huber objective function")

        # draw data points as short vertical lines
        plt.axvline(x = data[0][0], color = (1,0,0), ymax = 0.1, lw = 1.0, label="Inlier data values") # will be overwritten with corrected colour
        plt.axvline(x = data[0][0], color = (0,0,1), ymax = 0.1, lw = 1.0, label="Outlier data values") # will be overwritten with corrected colour
        max_weight = max(final_weight)
        for d,w in zip(data, final_weight, strict=True):
            alpha = w/max_weight
            color = [alpha, 0.0, 1.0-alpha]
            plt.axvline(x = d, color = color, ymax = 0.1, lw = 1.0)

        plt.axvline(x = m, color = 'limegreen', label = 'Estimated mean', lw = 1.0)
        plt.legend()
        plt.savefig(os.path.join(output_folder, "mean_pseudo_huber.png"), bbox_inches='tight')
        if not test_run:
            plt.show()

def mean_geman_mcclure_solver(data: np.array, scale: np.array, x_range: float, sigma_pop: float,
                              test_run: bool, output_folder: str) -> None:
    model_instance = LinearRegressor(data[0])
    p = 0.3
    sigma_base = sigma_pop/p
    sigma_limit = x_range
    num_sigma_steps = 20
    influence_func_instance = GemanMcClureInfluenceFunc(sigma=sigma_base)
    param_instance = GNC_WelschParams(influence_func_instance, sigma_base, sigma_limit, num_sigma_steps)
    irls_instance = IRLS(param_instance, data, model_instance=model_instance, print_warnings=False)
    if irls_instance.run():
        m = irls_instance.final_model[0]
        if not test_run:
            print("Geman-McClure IRLS result: m=", m)

    sup_gn_instance = SupGaussNewton(param_instance, data, model_instance=model_instance, print_warnings=False, debug=True)
    if sup_gn_instance.run():
        m = sup_gn_instance.final_model[0]
        final_weight = irls_instance.final_weight
        if not test_run:
            print("Geman-McClure Sup-GN result: m=", m)
            print("  final_weight=",final_weight)

    # check derivatives
    #residual = np.array([0.01])
    #rhop, Bterm = sup_gn_instance.calc_influence_func_derivatives(residual, 1.0) # scale
    #sup_gn_instance.numeric_derivs_model = True
    #rhopn, Btermn = sup_gn_instance.calc_influence_func_derivatives(residual, 1.0) # scale
    #if not test_run:
    #    print("rhop=",rhop, rhopn, "Bterm=",Bterm,Btermn)

    irls_instance = IRLS(param_instance, data, model_instance=model_instance, scale=scale)
    if irls_instance.run():
        mscale = irls_instance.final_model[0]
        if not test_run:
            print("Geman-McClure scale result difference=", mscale-m)

    if output_folder is not None:
        # create GNC instance for plotting
        param_instance_plot = GNC_WelschParams(GemanMcClureInfluenceFunc(), sigma_base, sigma_limit=sigma_limit,
                                               num_sigma_steps=n_intermediate_gnc_curves)
        sup_gn_instance_plot = SupGaussNewton(param_instance_plot, data, model_instance=model_instance)

        plot_gnc_example(sup_gn_instance_plot, m, x_range, final_weight, 0.0, # offset
                         "Geman-McClure", test_run, output_folder, "mean_geman_mcclure.png")
        
def mean_gnc_irls_p_solver(data: np.array, scale: np.array, x_range: float, sigma_pop: float,
                           test_run: bool, output_folder: str) -> None:
    p = 0.0
    rscale = 0.8
    epsilon_base = sigma_pop/0.6667
    epsilon_limit = 1.0
    beta = 0.95

    model_instance = LinearRegressor(data[0])
    influence_func_instance = GNC_IRLSpInfluenceFunc()
    param_instance = GNC_IRLSpParams(influence_func_instance, p, rscale, epsilon_base, epsilon_limit, beta)
    irls_instance = IRLS(param_instance, data, model_instance=model_instance, print_warnings=False, debug=True)
    if irls_instance.run():
        m = irls_instance.final_model[0]
        final_weight = irls_instance.final_weight
        if not test_run:
            print("GNC IRLS-p IRLS Result: m=", m)
            print("  final_weight=",final_weight)

    mean_finder = LinearRegressorGNC_IRLSp(p, rscale, epsilon_base, epsilon_limit, beta,
                                           print_warnings=False, debug=True)
    if mean_finder.run(data):
        m = mean_finder.final_intercept[0]
        final_weight = mean_finder.final_weight
        if not test_run:
            print("Pseudo-Huber Linear Regression optimisation result: m=", m)
            print("  final weights:",final_weight)

    # for checkout derivatives
    #print("rhop=",irlsInstance.updated_weight(np.array([2]),1.0,1.e-5))
    #irlsInstance.numeric_derivs_model = True
    #print("rhopn=",irlsInstance.updated_weight(np.array([2]),1.0,1.e-5))

    # for checking threshold handling
    #print("epsilon_base=",epsilon_base,"rscale=",rscale)
    #threshold = epsilon_base/rscale
    #print("threshold r=",threshold)
    #print("Just before: ",optimiser_instance.param_instance.influence_func_instance.rho(np.array([(threshold-0.00001)**2.0]), 1.0))
    #print("Just after: ",optimiser_instance.param_instance.influence_func_instance.rho(np.array([(threshold+0.00001)**2.0]), 1.0))
                                                                                    
    if output_folder is not None:
        # create GNC instance for plotting
        param_instance_plot = GNC_IRLSpParams(influence_func_instance, p, rscale, epsilon_base, epsilon_limit, 0.97) # beta
        sup_gn_instance_plot = SupGaussNewton(param_instance_plot, data, model_instance=model_instance)

        plot_gnc_example(sup_gn_instance_plot, m, x_range, final_weight, 0.0, # offset
                         "GNC IRLS-p0", test_run, output_folder, "mean_gnc_irls_p0.png")

def main(test_run:bool, output_folder:str="../../output"):
    np.random.seed(0) # We want the numbers to be the same on each run

    # data generation
    sigma_pop = 0.2 # population distribution standard deviation
    n_good_points = 10
    n_bad_points = 12
    mean_gt = 3.0
    x_range = 10.0
    data = np.zeros((n_good_points+n_bad_points,1))
    for i in range(n_good_points):
        data[i][0] = np.random.normal(mean_gt, sigma_pop)

    for i in range(n_good_points,n_good_points+n_bad_points):
        while(True):
            data[i][0] = x_range*np.random.rand()
            if abs(data[i][0]) > 3.0*sigma_pop:
                break

    scale = np.zeros(n_good_points+n_bad_points)
    scale[:] = 1.0

    mean_welsch_solver(data, scale, x_range, sigma_pop, test_run, output_folder)
    mean_pseudo_huber_solver(data, scale, x_range, sigma_pop, test_run, output_folder)
    mean_geman_mcclure_solver(data, scale, x_range, sigma_pop, test_run, output_folder)
    mean_gnc_irls_p_solver(data, scale, x_range, sigma_pop, test_run, output_folder)

    if test_run:
        print("mean_solver OK")

if __name__ == "__main__":
    main(True) # test_run
