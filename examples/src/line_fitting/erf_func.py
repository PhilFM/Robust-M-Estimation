import numpy as np
import math
import matplotlib.pyplot as plt

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

def erf_func(x):
    if abs(x) < 0.0001:
        return 2.0/math.sqrt(math.pi)
    else:
        return math.erf(x)/x

sqrt_2 = math.sqrt(2.0)

def erf_func2(a,b,Np,D,x,sigma):
    if abs(a) < 0.000001:
        a = 0.0001

    x1 = (a*D + b)/(sqrt_2*sigma)
    x2 = (a*x + b)/(sqrt_2*sigma)
    d = math.pi*Np*(math.erf(x1) - math.erf(x2))/(sqrt_2*D)
    return d/a

def normal_func(x):
    sigma = 2.0
    scale = 10.0
    return scale*math.exp(-0.5*x*x/(sigma*sigma))

def main(test_run:bool, output_folder:str="../../output"):
    x_range = 10.0
    xlist = np.linspace(-x_range, x_range, num=201)
    plt.figure(num=1, dpi=240)
    erf_mfv = np.vectorize(erf_func2, excluded={"b","Np","D","x","sigma"})
    normal_mfv = np.vectorize(normal_func)
    erf_list = erf_mfv(xlist, b=0.0, Np=10, D=5.0, x=-4.0, sigma=1.0)
    normal_list = normal_mfv(xlist)
    plt.plot(xlist, erf_list, lw = 1.0, label="erf2")
    plt.plot(xlist, normal_list, lw = 1.0, label="normal")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main(False) # test_run
