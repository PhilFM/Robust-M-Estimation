import math
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

sqrt_2 = math.sqrt(2.0)
sqrt_pi = math.sqrt(math.pi)
    
def main(test_run:bool, output_folder:str="../../../output"):
    output_folder += "/line_fit/breakdown_point"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    def sum_data_func(a:float, obs):
        tot = 0.0
        for i in range(len(obs)):
            for j in range(i+1,len(obs)):
                tot += abs(a - (obs[j][1] - obs[i][1])/(obs[j][0] - obs[i][0]))

        return tot

    def sum_approx_func(x:float, sigma:float, n_points:int, D:float):
        return n_points*D

    # build data
    alpha = 0.13
    n_points = 100
    n_bad_points = int(alpha*n_points)
    obs = np.zeros((n_points,2))
    sigma_pop = 1.0
    D = 5.0
    theta = 2.7372292427317007
    gamma = 2.66800562663619
    (a_bad,b_bad) = (gamma*math.cos(theta), gamma*math.sin(theta))
    for i in range(n_bad_points):
        obs[i][0] = -D + 2.0*D*i/(n_points-1)
        obs[i][1] = a_bad*obs[i][0]+b_bad

    for i in range(n_bad_points,n_points):
        obs[i][0] = -D + 2.0*D*i/(n_points-1)
        obs[i][1] = np.random.normal(0.0, sigma_pop)

    sum_data_func_v = np.vectorize(sum_data_func, excluded={"obs"})
    #sum_approx_func_v = np.vectorize(sum_approx_func, excluded={"sigma", "n_points", "D"})

    plt.close("all")
    plt.figure(num=1, dpi=240)
    ax = plt.gca()
    #ax.set_ylim((0.0, 1.4))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$F_G(x)$")

    n_a_vals = 201
    alist = np.linspace(-2.0, 2.0, n_a_vals)
    data_list = sum_data_func_v(alist, obs=obs)/(n_points*(n_points-1))
    a_best = alist[np.argmin(data_list)]
    #approx_list = sum_approx_func_v(alist, sigma=sigma_pop, n_points=n_points, D=D)

    small_diff = (alist[n_a_vals-1]-alist[0])/(n_a_vals-1)
    deriv_1 = 0.5*(data_list[2] - data_list[0])/small_diff
    if not test_run:
        print("first deriv is",deriv_1/(n_points*n_points))

    #for i in range(n_x_vals-4):
    #    deriv_1 = 0.5*(approx_list[i+3]-approx_list[i+1])/small_diff
    #    deriv_2 = (approx_list[i+1] - 2.0*approx_list[i+2] + approx_list[i+3])/(small_diff*small_diff)
    #    deriv_3 = 0.5*(approx_list[i+4] - 2.0*approx_list[i+3] + 2.0*approx_list[i+1] - approx_list[i])/(small_diff*small_diff*small_diff)
    #    print("derivs at",xlist[i+2],":",deriv_1/N,deriv_2/N,deriv_3/N)

    plt.axvline(x = a_best, label="Minimum data", color="cyan")
    plt.axvline(x = -sigma_pop/D, label="Limit on best $a$", color="red", linestyle="dotted")
    plt.plot(alist, data_list, lw = 1.0, color = 'blue')
    #plt.plot(alist, approx_list, lw = 1.0, color = 'red', linestyle = "dashed")

    plt.legend()
    plt.savefig(os.path.join(output_folder, "theil_sen_breakdown.png"), bbox_inches='tight')
    if not test_run:
        plt.show()

    if test_run:
        print("theil_sen_breakdown OK")

if __name__ == "__main__":
    main(False) # test_run
