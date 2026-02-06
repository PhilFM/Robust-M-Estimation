import numpy as np
import os
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    import sys
    sys.path.append("../../../pypi_package/src")
    sys.path.append("../../../pypi_package/src/gnc_smoothie/linear_model")
    sys.path.append("../../../pypi_package/src/gnc_smoothie/cython_files")

from gnc_smoothie.linear_model.linear_regressor_welsch import LinearRegressorWelsch

def plot_result(data, xy_range: float, final_mean_2d, test_run:bool, output_folder:str) -> None:
    plt.close("all")
    plt.figure(num=1, dpi=240)
    plt.clf()
    ax = plt.gca()
    ax.set_ylim(0.0,xy_range)
    ax.set_xlim(0.0,xy_range)
    ax.set_aspect('equal')

    for d in data:
        ax.plot(d[0][0], d[1][0], "o", markersize=2, color="b")

    ax.plot(data[0][0][0], data[0][1][0], "o", markersize=2, color="b", label="Data points")
    ax.plot(final_mean_2d[0], final_mean_2d[1], "o", color="g", markersize=3, label="Result mean")

    plt.legend()
    plt.savefig(os.path.join(output_folder, "mean_2d_example.png"), bbox_inches='tight')
    if not test_run:
        plt.show()

def solve_time(dim: int,
               test_run: bool,
) -> float:
    # data is a list of [x,y,z] triplets
    mean_gt = np.zeros(dim)
    range_gt = np.zeros(dim)
    for d in range(dim):
        mean_gt[d] = np.random.rand()
        range_gt[d] = 0.1 + np.random.rand()

    sigma_pop = 0.03
    n_good_points = 100
    n_bad_points = 10
    data = np.zeros((n_good_points+n_bad_points,dim,1))
    for i in range(n_good_points):
        for d in range(dim):
            data[i][d][0] = mean_gt[d] + np.random.normal(0.0, sigma_pop)

    for i in range(n_bad_points):
        for d in range(dim):
            data[n_good_points+i][d][0] = np.random.rand()*range_gt[d]

    start_time = time.time()            
    q = 0.6667
    sigma = sigma_pop/q
    sigma_limit = max(range_gt)
    linear_regressor = LinearRegressorWelsch(sigma, sigma_limit=sigma_limit, num_sigma_steps=20, use_slow_version=False, debug=True,
                                             max_niterations=5000) #, messages_file=sys.stdout)
    if linear_regressor.run(data):
        intercept = linear_regressor.final_model
        final_mean = np.array(intercept)
        debug_model_list = linear_regressor.debug_model_list

    elapsed_time = time.time() - start_time
    if not test_run:
        print("Linear regression dim=",dim,"time=", elapsed_time,"RMS error: ", np.sqrt((final_mean-mean_gt) ** 2).mean())
        #plot_result(data, xy_range, final_mean_2d, test_run, output_folder)

    return elapsed_time

def main(test_run:bool, output_folder:str="../../../output", quick_run:bool=False):
    np.random.seed(0) # We want the numbers to be the same on each run

    dim_time = []
    dim_list = range(1,20 if quick_run else 250)
    for dim in dim_list:
        dim_time.append(solve_time(dim, test_run))

    plt.close("all")
    plt.figure(num=1, dpi=240)
    ax = plt.gca()
    ax.set_xlabel(r"Dimension")
    ax.set_ylabel(r"Time in seconds")
    plt.plot(dim_list, dim_time)

    if test_run:
        plt.savefig(os.path.join(output_folder, "mean_nd_complexity_test.png"), bbox_inches='tight')
    else:
        plt.savefig(os.path.join(output_folder, "mean_nd_complexity.png"), bbox_inches='tight')
        plt.show()

    if test_run:
        print("mean_nd_complexity OK")

if __name__ == "__main__":
    main(False) # test_run
