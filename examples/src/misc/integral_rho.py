import math
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

    
# checks integral used for S-estimation: Expected value of rho(r^2) = exp(-r^2/(2*sigma^2)) for Welsch influence function, when sigma=1 and r is N(0,1)
# We have E(rho(r^2)) = integral_-infty^+infty exp(-r^2/2)*exp(-r^2/2) dr / sqrt(2*pi)
#                     = sqrt(pi)*erf(infty) / sqrt(2*pi)
#                     = sqrt(1/2)
def integrate_welsch(data:np.ndarray, test_run:bool, output_folder:str):
    plt.close("all")
    plt.figure(num=1, dpi=120)

    #ax = plt.gca()
    #ax.set_xscale("log")
    #ax.set_yscale("log")

    sigma_list = []
    sigma_est_list = []
    for sigma in np.linspace(1.0,10.0,10):
        tot = 0.0
        for i in range(len(data)):
            tot += math.exp(-0.5*data[i]*data[i]/(sigma*sigma))

        av = tot / len(data)
        sigma_list.append(sigma)
        sigma_est = math.sqrt(av*av/(1.0-av*av))
        sigma_est_list.append(sigma_est)
        if not test_run:
            print("sigma=",sigma,"sigma_est=",sigma_est)

    plt.plot(sigma_list,sigma_est_list,marker="o",color="green")

    #plt.legend()
    plt.savefig(os.path.join(output_folder, "integral_rho.png"), bbox_inches='tight')
    if not test_run:
        plt.show()

# integrates Tukey bisquare: Expected value of rho(r^2) = exp(-r^2/(2*sigma^2)) for Tukey Bisquare influence function,
# for c=1.547 and r is N(0,1)
def integrate_tukey_bisquare(data:np.ndarray, test_run:bool, output_folder:str):
    def rho(r:float, c:float):
        return c**2 / 6 if abs(r) >= c else (r**2 / 2) - (r**4 / (2 * c**2)) + (r**6 / (6 * c**4))

    for c in [1.547,3.44,4.685]:
        tot = 0.0
        for i in range(len(data)):
            tot += rho(data[i], c)

        av = tot / len(data)
        if not test_run:
            print("c=",c,"s=",av)

def main(test_run:bool, output_folder:str="../../../output"):
    output_folder += "/misc"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    n_points = 10000 #000
    data = np.zeros(n_points)
    for i in range(n_points):
        data[i] = np.random.normal(0.0, 1.0)

    integrate_welsch(data, test_run, output_folder)
    integrate_tukey_bisquare(data, test_run, output_folder)

    if test_run:
        print("integral_rho OK")

if __name__ == "__main__":
    main(False) # test_run
