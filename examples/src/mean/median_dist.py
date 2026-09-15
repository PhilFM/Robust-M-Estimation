import math
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

sqrt_2 = math.sqrt(2.0)
sqrt_pi = math.sqrt(math.pi)
    
def main(test_run:bool, output_folder:str="../../../output"):
    def sum_abs_diff_func(x:float, obs):
        tot = 0.0
        for z in obs:
            tot += abs(x-z)

        return tot

    def sum_abs_diff_approx_func(x:float, mean_gt:float, sigma:float, N:int):
        xd = x - mean_gt
        return N*(xd*math.erf(xd/(sqrt_2*sigma)) + sqrt_2*sigma*math.exp(-0.5*xd*xd/(sigma*sigma))/sqrt_pi)

    # build data
    N = 10000
    obs = np.zeros(N)
    sigma_pop = 1.4
    mean_gt = 5.0
    for i in range(N):
        obs[i] = np.random.normal(mean_gt, sigma_pop)

    sum_abs_diff_func_v = np.vectorize(sum_abs_diff_func, excluded={"obs"})
    sum_abs_diff_approx_func_v = np.vectorize(sum_abs_diff_approx_func, excluded={"mean_gt", "sigma", "N"})

    ax = plt.gca()
    #ax.set_ylim((0.0, 1.4))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$F_G(x)$")

    n_x_vals = 201
    xlist = np.linspace(0.0, 10.0, n_x_vals)
    diff_list = sum_abs_diff_func_v(xlist, obs=obs)
    approx_list = sum_abs_diff_approx_func_v(xlist, mean_gt=mean_gt, sigma=sigma_pop, N=N)

    small_diff = (xlist[n_x_vals-1]-xlist[0])/(n_x_vals-1)
    for i in range(n_x_vals-4):
        deriv_1 = 0.5*(approx_list[i+3]-approx_list[i+1])/small_diff
        deriv_2 = (approx_list[i+1] - 2.0*approx_list[i+2] + approx_list[i+3])/(small_diff*small_diff)
        deriv_3 = 0.5*(approx_list[i+4] - 2.0*approx_list[i+3] + 2.0*approx_list[i+1] - approx_list[i])/(small_diff*small_diff*small_diff)
        if not test_run:
            print("derivs at",xlist[i+2],":",deriv_1/N,deriv_2/N,deriv_3/N)

    plt.plot(xlist, diff_list, lw = 1.0, color = 'blue')
    plt.plot(xlist, approx_list, lw = 1.0, color = 'red', linestyle = "dashed")

    #plt.legend()
    plt.savefig(os.path.join(output_folder, "median_dist.png"), bbox_inches='tight')
    if not test_run:
        plt.show()

    if test_run:
        print("median_dist OK")

if __name__ == "__main__":
    main(False) # test_run
