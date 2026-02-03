import numpy as np
import matplotlib.pyplot as plt
import os
import time

if __name__ == "__main__":
    import sys
    sys.path.append("../../../pypi_package/src")

# Welsch
from gnc_smoothie_philfm.linear_model.linear_regressor_welsch import LinearRegressorWelsch

def mean_welsch_solver(data: np.array, scale: np.array, x_min: float, x_max: float, sigma_pop: float,
                       test_run: bool, output_folder: str) -> None:
    # estimation parameters
    q = 0.66667 # ratio of population standard deviation to base sigma value in estimation
    sigma_base = sigma_pop/q
    sigma_limit = x_max-x_min
    num_sigma_steps = 20
    max_niterations = 500

    mean_finder = LinearRegressorWelsch(sigma_base, sigma_limit=sigma_limit, num_sigma_steps=num_sigma_steps,
                                        max_niterations=max_niterations, messages_file=None, debug=True)
    if mean_finder.run(data):
        m = mean_finder.final_model
        if not test_run:
            print("Welsch Sup-GN optimisation result: m=", m)
            #print("  final weights:",final_weight)

def main(test_run:bool, output_folder:str="../../../output"):
    np.random.seed(0) # We want the numbers to be the same on each run

    # data generation
    sigma_pop = 0.2 # population distribution standard deviation
    n_dimensions = 100
    n_good_points = 500
    n_bad_points = 200
    mean_gt = 3.0
    x_min = 0.0
    x_max = 10.0
    data = np.zeros((n_good_points+n_bad_points,n_dimensions,1))
    for i in range(n_good_points):
        for d in range(n_dimensions):
            data[i][d][0] = np.random.normal(mean_gt, sigma_pop)

    for i in range(n_good_points,n_good_points+n_bad_points):
        for d in range(n_dimensions):
            while(True):
                data[i][d][0] = x_max*np.random.rand()
                if abs(data[i][d][0]) > 3.0*sigma_pop:
                    break

    scale = np.zeros(len(data))
    scale[:] = 1.0

    time_list = []
    for d in range(n_dimensions):
        datap = data[:,:d+1,:]
        print(datap.shape)
        start_time = time.time()
        mean_welsch_solver(datap, scale, x_min, x_max, sigma_pop, test_run, output_folder)
        time_list.append(time.time() - start_time)

    plt.close("all")
    plt.figure(num=1, dpi=240)
    plt.plot(time_list)
    plt.savefig(os.path.join(output_folder, "mean_complexity.png"), bbox_inches='tight')
    if not test_run:
        plt.show()

    if test_run:
        print("mean_solver OK")

if __name__ == "__main__":
    main(False) # test_run
