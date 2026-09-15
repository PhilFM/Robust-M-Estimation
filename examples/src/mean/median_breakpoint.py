import numpy as np
import matplotlib.pyplot as plt
import math
import os
from pathlib import Path

sqrt_2 = math.sqrt(2.0)
inv_sqrt_2 = 1.0/sqrt_2
sqrt_pi = math.sqrt(math.pi)

def sum_abs_diff_func(x:float, data:np.ndarray):
    tot = 0.0
    for z in data:
        tot += abs(x-z[0])

    return tot

def F_good(x:float, alpha:float, sigma:float):
    return (0.5*sqrt_pi)*sqrt_2*(1.0-alpha)*(sqrt_2*x*math.erf(x/(sqrt_2*sigma)) + (2.0/sqrt_pi)*sigma*math.exp(-0.5*x*x/(sigma*sigma)))/sqrt_pi

def F_bad(x:float, alpha:float, x_bad:float):
    return alpha*abs(x-x_bad)

def F_tot(x:float, alpha:float, sigma:float, x_bad:float):
    return F_good(x,alpha,sigma) + F_bad(x,alpha,x_bad)

def main(test_run:bool, output_folder:str="../../../output", quick_run:bool=False):
    output_folder += "/mean/breakdown_point"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    np.random.seed(0) # We want the numbers to be the same on each run

    # data generation
    sigma_pop = 0.2 # population distribution standard deviation
    alpha = 0.49
    n_points = 50000
    n_bad_points = int(alpha*n_points)
    n_good_points = n_points - n_bad_points
    mean_gt = 0.0
    for x_bad in np.linspace(0.0, 0.5, 2) if quick_run else np.linspace(0.0, 5.0, 21):
        data = np.zeros((n_points,1))
        for i in range(n_good_points):
            data[i][0] = np.random.normal(mean_gt, sigma_pop)

        for i in range(n_good_points,n_good_points+n_bad_points):
            data[i][0] = x_bad

        x_list = np.linspace(-0.1,2,201)
        hmfv_d = np.vectorize(sum_abs_diff_func, excluded={"data"})
        hmfv_p = np.vectorize(F_tot, excluded={"alpha", "sigma", "x_bad"})

        plt.close("all")
        plt.figure(num=1, dpi=240)
        ax = plt.gca()
        #plt.box(False)
        #ax.set_xlim((x_min, x_max))
        d_list = hmfv_d(x_list, data=data)/n_points
        p_list = hmfv_p(x_list, alpha=alpha, sigma=sigma_pop, x_bad=x_bad)
        x_best = x_list[np.argmin(p_list)]
        if not test_run:
            print("Best ratio=",x_best/sigma_pop)

        ax.set_ylim(0.0, 1.05*max(max(d_list),max(p_list)))

        plt.axvline(x = x_best, label="Minimum prediction")
        plt.plot(x_list, d_list, label="data") #, lw = 1.0, color=color)
        plt.plot(x_list, p_list, label="predicted", linestyle="dotted") #, lw = 1.0, color=color)
        plt.legend()
        plt.savefig(os.path.join(output_folder, "median_breakpoint_xb="+str(x_bad)+".png"), bbox_inches='tight')
        if not test_run:
            plt.show()

    if test_run:
        print("median_breakpoint OK")

if __name__ == "__main__":
    main(False) # test_run
