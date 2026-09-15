import math
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

def main(test_run:bool, output_folder:str="../../../output"):
    output_folder += "/misc"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    rlist = np.linspace(-3.0, 3.0, num=100)
    plt.close("all")
    plt.figure(num=1, dpi=240)

    def sum_normal_func(x:float, mu_arr:np.ndarray):
        tot = 0.0
        for mu in mu_arr:
            tot += math.exp(-0.5*(x-mu)*(x-mu))

        return tot

    sumGauFuncV = np.vectorize(sum_normal_func, excluded={"mu_arr"})

    mu_arr = [0.1, 0.3, 0.9]

    ax = plt.gca()
    ax.set_xlim((-0.6, 1.6))
    ax.set_ylim((0.0, 3.1))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"Sy$")

    plt.axvline(x = 0.0, color = 'magenta', lw = 1.0, linestyle = "dashed", label=r"$x=0$")
    plt.axvline(x = 1.0, color = 'cyan', lw = 1.0, linestyle = "dashed", label=r"$x=\sigma$")
    plt.axvline(x = mu_arr[0], color = 'red', ymax = 0.3333, lw = 1.0, linestyle = "dotted", label = "Offsets of each Gaussian")
    for mu in mu_arr:
        plt.axvline(x = mu, color = 'red', ymax = 0.3333, lw = 1.0, linestyle = "dotted")

    plt.plot(rlist, sumGauFuncV(rlist, mu_arr=[mu_arr[0]]), lw = 1.0, color = 'blue', label = "Individual Gaussians")
    for mu in mu_arr:
        plt.plot(rlist, sumGauFuncV(rlist, mu_arr=[mu]), lw = 1.0, color = 'blue')

    plt.plot(rlist, sumGauFuncV(rlist, mu_arr=mu_arr), lw = 1.0, color = 'green', label = "Sum of Gaussians") #, label="erf div test")

    plt.legend()
    plt.savefig(os.path.join(output_folder, "sum_of_gaussians.png"), bbox_inches='tight')
    if not test_run:
        plt.show()

    if test_run:
        print("erf_check OK")

if __name__ == "__main__":
    main(False) # test_run
