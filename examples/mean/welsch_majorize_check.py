import math
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

def main(test_run:bool, output_folder:str="../../output"):
    rlist = np.linspace(-3.0, 3.0, num=100)
    plt.close("all")
    plt.figure(num=1, dpi=240)

    def welsch_majorize_func(x):
        return -1.0 + math.exp(x) - x

    majFuncV = np.vectorize(welsch_majorize_func)

    ax = plt.gca()
    ax.set_ylim((0.0, 16.0))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$f(x)$")

    plt.plot(rlist, majFuncV(rlist), lw = 1.0, color = 'green') #, label="Welsch majorize test")

    #plt.legend()
    plt.savefig(os.path.join(output_folder, "welsch_majorize_check.png"), bbox_inches='tight')
    if not test_run:
        plt.show()

    if test_run:
        print("majorize OK")

if __name__ == "__main__":
    main(False) # test_run
