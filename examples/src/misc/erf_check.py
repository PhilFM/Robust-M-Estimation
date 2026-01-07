import math
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import brentq

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

def main(test_run:bool, output_folder:str="../../output"):
    rlist = np.linspace(-3.0, 3.0, num=100)
    plt.close("all")
    plt.figure(num=1, dpi=240)

    # scale normal distribution vertically to erf(x)/x at zero, and change width to match curvature
    # if we have g(x) = 2*exp(-k*x^2)/sqrt(pi) then 2nd derivative is -4*k/sqrt(pi), and 2nd derivative
    # of erf(x)/x is -4/(3*sqrt(pi)), so we have
    #  -4*k/sqrt(pi) = -4/(3*sqrt(pi)) so k=1/3
    def normal_func(x):
        return 2.0*math.pow(math.pi,-0.5)*math.exp(-0.33333333*x*x)

    def erf_div_func(x):
        if abs(x) < 0.000001:
            return 2.0*math.pow(math.pi,-0.5)
        else:
            return math.erf(x)/x

    def erf_div_2nd_deriv(x):
        if abs(x) < 0.0001:
            return -4.0/(3.0*math.sqrt(math.pi))
        else:
            return 2.0*(math.erf(x) - 2.0*math.exp(-x*x)*x*(1.0 + x*x)/math.sqrt(math.pi))/(x*x*x)

    def erf_div_deriv(x):
        return 2.0*math.exp(-x*x)/(math.sqrt(math.pi)*x) - math.erf(x)/(x*x)

    def erf_deriv(x):
        return 2.0*math.exp(-x*x)/math.sqrt(math.pi)

    gauFuncV = np.vectorize(normal_func)
    erfFuncV = np.vectorize(erf_div_func)

    ax = plt.gca()
    ax.set_ylim((0.0, 1.4))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$f(x)$")

    small_val = 0.0001
    x = 0.1
    if not test_run:
        print("grad erf num=", 0.5*(math.erf(x+small_val) - math.erf(x-small_val))/small_val, " (",erf_deriv(x),")")
        print("grad erf div num=", 0.5*(erf_div_func(x+small_val) - erf_div_func(x-small_val))/small_val, " (",erf_div_deriv(x),")")
        print("normal 2nd deriv=", (normal_func(small_val) + normal_func(-small_val) - 2.0*normal_func(0.))/(small_val*small_val))
        print("erf div 2nd deriv=", (erf_div_func(x+small_val) + erf_div_func(x-small_val) - 2.0*erf_div_func(x))/(small_val*small_val), " (", erf_div_2nd_deriv(x),")",erf_div_2nd_deriv(x)*math.sqrt(math.pi)*0.75,-4.0/(3.0*math.sqrt(math.pi)))

    xsol = brentq(erf_div_2nd_deriv, -0.0001, -5000.0)
    if not test_run:
        print("erf div 2nd deriv zero point =",xsol)

    plt.plot(rlist, gauFuncV(rlist), lw = 1.0, color = 'blue')
    plt.plot(rlist, erfFuncV(rlist), lw = 1.0, color = 'green') #, label="erf div test")

    #plt.legend()
    plt.savefig(os.path.join(output_folder, "erf_check.png"), bbox_inches='tight')
    if not test_run:
        plt.show()

    if test_run:
        print("erf_check OK")

if __name__ == "__main__":
    main(False) # test_run
