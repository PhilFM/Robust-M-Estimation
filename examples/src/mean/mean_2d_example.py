import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    import sys
    sys.path.append("../../../pypi_package/src")
    sys.path.append("../../../pypi_package/src/gnc_smoothie_philfm/linear_model")
    sys.path.append("../../../pypi_package/src/gnc_smoothie_philfm/cython_files")

from gnc_smoothie_philfm.linear_model.linear_regressor_welsch import LinearRegressorWelsch

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

def main(test_run:bool, output_folder:str="../../../output"):
    np.random.seed(0) # We want the numbers to be the same on each run

    # data is a list of [x,y,z] triplets
    mean_gt = [0.5, 0.3]

    sigma_pop = 0.03
    xy_range = 1.2
    n_good_points = 20
    n_bad_points = 50
    data = np.zeros((n_good_points+n_bad_points,2,1))
    for i in range(n_good_points):
        data[i][0][0] = mean_gt[0] + np.random.normal(0.0, sigma_pop)
        data[i][1][0] = mean_gt[1] + np.random.normal(0.0, sigma_pop)

    for i in range(n_bad_points):
        data[n_good_points+i][0][0] = np.random.rand()*xy_range
        data[n_good_points+i][1][0] = np.random.rand()*xy_range
    
    q = 0.6667
    sigma = sigma_pop/q
    sigma_limit = xy_range
    linear_regressor = LinearRegressorWelsch(sigma, sigma_limit=sigma_limit, num_sigma_steps=20, use_slow_version=False, debug=True)
    if linear_regressor.run(data):
        intercept = linear_regressor.final_model
        final_mean_2d = np.array([intercept[0], intercept[1]])
        debug_model_list = linear_regressor.debug_model_list

    if not test_run:
        print("Linear regression 2D mean result:", final_mean_2d)
        print("   error: ", final_mean_2d-mean_gt)
        plot_result(data, xy_range, final_mean_2d, test_run, output_folder)

    # change to True if you want to see the progress of the algorithm
    if False: #not test_run:
        print("Intermediate 2D mean model values:")
        for m in debug_model_list:
            if not test_run:
                print("   ",m)

    if test_run:
        print("mean_2d_example OK")

if __name__ == "__main__":
    main(False) # test_run
