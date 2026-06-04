import math
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

sigma = 1.0

def quadratic_influence(x: float):
    return 0.5*x*x/(sigma*sigma)

def huber_influence(x: float):
    delta = 1.0
    if abs(x) < delta:
        return 0.5*x*x
    else:
        return delta*(abs(x) - 0.5*delta)

def welsch_influence(x: float):
    return 1.0-math.exp(-0.5*x*x/(sigma*sigma))

def main(test_run:bool, output_folder:str="../../../output"):
    rlist = np.linspace(-2.5, 2.5, num=100)
    plt.close("all")
    plt.figure(num=1, dpi=240)

    qFuncV = np.vectorize(quadratic_influence)
    wFuncV = np.vectorize(welsch_influence)
    hFuncV = np.vectorize(huber_influence)

    ax = plt.gca()
    #ax.set_ylim((0.0, 16.0))
    ax.set_xlabel(r"Residual error")
    ax.set_ylabel(r"Loss function")

    plt.plot(rlist, qFuncV(rlist), lw = 1.0, label="Quadratic loss")
    plt.plot(rlist, hFuncV(rlist), lw = 1.0, label="Huber loss")
    plt.plot(rlist, wFuncV(rlist), lw = 1.0, label="Welsch loss")

    plt.legend()
    plt.savefig(os.path.join(output_folder, "compare_influence.png"), bbox_inches='tight')
    if not test_run:
        plt.show()

    if test_run:
        print("compare_influence OK")

if __name__ == "__main__":
    main(False) # test_run
