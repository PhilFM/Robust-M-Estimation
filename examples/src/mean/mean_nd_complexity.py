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
from gnc_smoothie.irls import IRLS
from gnc_smoothie.cython_files.linear_regressor_welsch_evaluator import LinearRegressorWelschEvaluator
from gnc_smoothie.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie.welsch_influence_func import WelschInfluenceFunc

def solve_time(dim: int,
               outlier_fraction: float,
               file_id: str,
               test_run: bool,
) -> float:
    # data is a list of [x,y,z] triplets
    mean_gt = np.zeros(dim)
    range_gt = np.zeros(dim)
    for d in range(dim):
        range_gt[d] = 20.0 + 80.0*np.random.rand()
        mean_gt[d] = np.random.rand()*range_gt[d]

    sigma_pop = 1.0
    n_points = 200
    n_bad_points = int(outlier_fraction*n_points)
    n_good_points = n_points - n_bad_points
    data = np.zeros((n_points,dim,1))
    for i in range(n_good_points):
        for d in range(dim):
            data[i][d][0] = mean_gt[d] + np.random.normal(0.0, sigma_pop)

    for i in range(n_bad_points):
        for d in range(dim):
            data[n_good_points+i][d][0] = np.random.rand()*range_gt[d]

    start_time = time.time()            
    q = 0.3 #0.6667
    sigma = sigma_pop/q
    sigma_limit = max(np.max(np.max(data,0)-np.min(data,0)),max(range_gt))
    #print("sigma_limit=",sigma_limit)
    num_sigma_steps = 100 #20
    max_niterations = 5000
    if file_id == "supgn":
        linear_regressor = LinearRegressorWelsch(sigma, sigma_limit=sigma_limit, num_sigma_steps=num_sigma_steps, use_slow_version=False, debug=False,
                                                 max_niterations=max_niterations) #, messages_file=sys.stdout)
        if linear_regressor.run(data):
            intercept = linear_regressor.final_model
            final_mean = np.array(intercept)
    elif file_id == "irls":
        irls_instance = IRLS(GNC_WelschParams(WelschInfluenceFunc(), sigma,
                                              sigma_limit=sigma_limit,  num_sigma_steps=num_sigma_steps),
                             data,
                             evaluator_instance=LinearRegressorWelschEvaluator(data[0]),
                             max_niterations=max_niterations)
        if irls_instance.run():
            final_mean = irls_instance.final_model
            #print("final_weight:",irls_instance.final_weight)
    elif file_id == "average":
        final_mean = np.mean(data.reshape((n_points,dim)), axis=0)
        #print("final_mean=",final_mean)

    rms_error = np.sqrt((final_mean-mean_gt) ** 2).mean()
    elapsed_time = time.time() - start_time
    return rms_error,elapsed_time

def calc_complexity(
        outlier_fraction_list: list[float],
        dim_list: list[int],
        n_samples: int,
        file_id: str,
        test_run: bool,
        output_folder: str,
        quick_run: bool
) -> None:
    rms_error_median_list = []
    for outlier_fraction in outlier_fraction_list:
        rms_error_list = []
        dim_time_list = []
        for dim in dim_list:
            tot_rms_error = 0.0
            tot_elapsed_time = 0.0
            for sample in range(n_samples):
                rms_error,elapsed_time = solve_time(dim, outlier_fraction, file_id, test_run)
                tot_rms_error += rms_error
                tot_elapsed_time += elapsed_time

            if not test_run:
                print("Outlier fraction=",outlier_fraction,"dim=",dim,"time=", elapsed_time,"RMS error=", rms_error)

            rms_error_list.append(tot_rms_error/n_samples)
            dim_time_list.append(tot_elapsed_time/n_samples)

        rms_error_median_list.append(np.median(rms_error_list))

        plt.close("all")
        plt.figure(num=1, dpi=240)
        ax = plt.gca()
        ax.set_xlabel(r"Dimension")
        ax.set_ylabel(r"Time in seconds")
        plt.plot(dim_list, dim_time_list)

        if test_run:
            plt.savefig(os.path.join(output_folder, "mean_nd_complexity_" + file_id + "-" + str(outlier_fraction) + "_test.png"), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(output_folder, "mean_nd_complexity_" + file_id + "-" + str(outlier_fraction) + ".png"), bbox_inches='tight')
            #plt.show()

        plt.close("all")
        plt.figure(num=1, dpi=240)
        ax = plt.gca()
        ax.set_xlabel(r"Dimension")
        ax.set_ylabel(r"RMS error")
        plt.plot(dim_list, rms_error_list)

        if test_run:
            plt.savefig(os.path.join(output_folder, "mean_nd_rms_error_" + file_id + "-" + str(outlier_fraction) + "_test.png"), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(output_folder, "mean_nd_rms_error_" + file_id + "-" + str(outlier_fraction) + ".png"), bbox_inches='tight')
            #plt.show()

    plt.close("all")
    plt.figure(num=1, dpi=240)
    ax = plt.gca()
    ax.set_xlabel(r"Outlier percentage")
    ax.set_ylabel(r"RMS error median")
    plt.plot(100.0*np.array(outlier_fraction_list), rms_error_median_list)

    if test_run:
        plt.savefig(os.path.join(output_folder, "mean_nd_rms_error_median_" + file_id + "_test.png"), bbox_inches='tight')
    else:
        plt.savefig(os.path.join(output_folder, "mean_nd_rms_error_median_" + file_id + ".png"), bbox_inches='tight')
        plt.show()

def main(test_run:bool, output_folder:str="../../../output", quick_run:bool=False):
    np.random.seed(0) # We want the numbers to be the same on each run

    outlier_fraction_list = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    dim_list = range(1,10) if quick_run else range(2,100)

    n_samples = 2 if quick_run else 10
    #calc_complexity(outlier_fraction_list, dim_list, n_samples, "average", test_run, output_folder, quick_run)
    calc_complexity(outlier_fraction_list, dim_list, n_samples, "irls", test_run, output_folder, quick_run)
    #calc_complexity(outlier_fraction_list ,dim_list, n_samples, "supgn", test_run, output_folder, quick_run)

    if test_run:
        print("mean_nd_complexity OK")

if __name__ == "__main__":
    main(False) # test_run
