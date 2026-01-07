import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    import sys
    sys.path.append("../../../pypi_package/src")

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.linear_model.linear_regressor import LinearRegressor

# Welsch
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc

def objective_func(m, optimiser_instance):
    return optimiser_instance.objective_func([m])

def get_y_limits(mlist: list, optimiser_instance) -> (float,float):
    # get min and max of data
    y_min = y_max = 0.0
    for mx in mlist:
        y_max = max(y_max, objective_func(mx, optimiser_instance))

    y_min *= 1.05 # allow for a small border
    y_max *= 1.05 # allow for a small border
    return y_min,y_max

def main(test_run:bool, output_folder:str="../../../output"):
    np.random.seed(0) # We want the numbers to be the same on each run

    # data generation
    sigma_pop = 0.2 # population distribution standard deviation
    n_points = 3
    x_offset = 3.0
    x_range = 10.0
    data = np.zeros((n_points,1))
    for i in range(n_points):
        while True:
            data[i][0] = x_offset + x_range*np.random.rand()
            close = False
            for j in range(i):
                if abs(data[i][0]-data[j][0]) < 0.5:
                    close = True

            if not close:
                break

    # sort data so that we can easily find small and large items
    data = np.sort(data, axis=0)
            
    # estimation parameters
    p = 0.66667 # ratio of population standard deviation to base sigma value in estimation
    sigma_base = sigma_pop/p
    sigma_limit = data[n_points-1][0] - data[0][0]
    num_sigma_steps = 20

    model_instance = LinearRegressor(data[0])
    param_instance = GNC_WelschParams(WelschInfluenceFunc(), sigma_base, sigma_limit=sigma_limit,
                                      num_sigma_steps=num_sigma_steps)
    sup_gn_instance = SupGaussNewton(param_instance, data, model_instance=model_instance)
    x_border = 0.5*(data[n_points-1][0] - data[0][0])
    x_min = data[0][0]          - x_border
    x_max = data[n_points-1][0] + x_border
    mlist = np.linspace(x_min, x_max, num=300)
    sup_gn_instance._param_instance.influence_func_instance.sigma = sigma_limit
    (y_min,y_max) = get_y_limits(mlist, sup_gn_instance)

    plt.close("all")
    plt.figure(num=1, dpi=240)
    ax = plt.gca()
    #plt.box(False)
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))

    hmfv = np.vectorize(objective_func, excluded={"optimiser_instance"})

    # example gaussians for individual data points
    for i in range(n_points):
        data_point = [data[i]]
        data_point_instance = SupGaussNewton(param_instance, data_point, model_instance=model_instance)
        plt.plot(mlist, hmfv(mlist, optimiser_instance=data_point_instance), lw = 1.0, color="limegreen", label="Data point contribution" if i==0 else None)
        
    plt.plot(mlist, hmfv(mlist, optimiser_instance=sup_gn_instance), lw = 1.0, color="green", label="Objective function")

    plt.axvline(x = data[0][0],          color = 'lightgrey', ymax = y_max, lw = 1.0, label="Data limits")
    plt.axvline(x = data[n_points-1][0], color = 'lightgrey', ymax = y_max, lw = 1.0)

    for d in data:
        ax.plot([d[0], d[0]], [0, 1.0], lw = 1.0, color = "b")

    plt.axvline(x = d[0], color = 'b', ymax = 0.1, lw = 1.0, label="Data values")

    plt.legend()
    plt.savefig(os.path.join(output_folder, "mean_limit.png"), bbox_inches='tight')
    if not test_run:
        plt.show()

    if test_run:
        print("mean_limit OK")

if __name__ == "__main__":
    main(False) # test_run
