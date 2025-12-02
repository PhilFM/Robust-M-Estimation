import math
import numpy as np
import matplotlib.pyplot as plt
import argparse

def refFunc(r):
    return 1.0 - math.exp(-0.5*r*r)

def majFunc(r):
    return 0.5*r*r*math.exp(-0.5*r*r)

def main(testrun:bool):
    rlist = np.linspace(-3.0, 3.0, num=100)
    refFuncV = np.vectorize(refFunc)
    majFuncV = np.vectorize(majFunc)
    plt.figure(num=1, dpi=240)

    ax = plt.gca()

    ax.set_ylim((0.0, 1.0))
    #ax.set_ylabel("f(y)")
    ax.set_xlabel("y")

    plt.plot(rlist, rlist*0, "--", label="")
    plt.plot(rlist, refFuncV(rlist), lw = 1.0, color = 'blue', label="Ref")
    plt.plot(rlist, majFuncV(rlist), lw = 1.0, color = 'green', label="Maj")

    plt.legend()
    plt.savefig("../../Output/majorize.png", bbox_inches='tight')
    if not testrun:
        plt.show()

    def welsch_majorize_func(x):
        return -1.0 + math.exp(x) - x

    majFuncV = np.vectorize(welsch_majorize_func)

    ax = plt.gca()
    ax.set_ylim((0.0, 16.0))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$f(x)$")

    plt.plot(rlist, majFuncV(rlist), lw = 1.0, color = 'green') #, label="Welsch majorize test")

    plt.legend()
    plt.savefig('../../Output/welsch_majorize_test.png', bbox_inches='tight')
    if not testrun:
        plt.show()

    if testrun:
        print("OK")

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--testrun', action="store_true", default=False)
args = parser.parse_args()
main(args.testrun)
