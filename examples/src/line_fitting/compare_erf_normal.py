# Compares erf(x)/x with Gaussian function
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

def erf_func(x):
    if abs(x) < 0.0001:
        return 2.0/math.sqrt(math.pi)
    else:
        return math.erf(x)/x

sqrt_2 = math.sqrt(2.0)

def erf_func2(a,b,x1,x2):
    if abs(a) < 0.000001:
        a = 0.0001

    d = math.erf(a*x2+b) - math.erf(a*x1+b)
    return d/a

def normal_func(x,sigma,scale):
    #print("x=",x,"sigma=",sigma,"scale=",scale,math.exp(-0.5*x*x/(sigma*sigma)))
    val = scale*math.exp(-0.5*x*x/(sigma*sigma))
    if abs(val) < 1.0e-6:
        val = 1.0e-6

    return val

def main(test_run:bool, output_folder:str="../../output"):
    output_folder += "/line_fit"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    x_range = 10.0
    xlist = np.linspace(-x_range, x_range, num=401)
    plt.close("all")
    plt.figure(num=1, dpi=240)
    erf_mfv_a = np.vectorize(erf_func2, excluded={"b","x1","x2"})
    normal_mfv = np.vectorize(normal_func, excluded={"sigma", "scale"})
    ratio_init = None
    x1 = -4.0
    x2 =  5.0
    for b in np.linspace(0.0, 5.0, num=21):
        erf_list_a = erf_mfv_a(xlist, b=b, x1=x1, x2=x2)
        normal_list = normal_mfv(xlist, sigma=1.0/(sqrt_2*x1), scale=1.0)
        erf_arr_a = np.array(erf_list_a)
        normal_arr = np.array(normal_list)
        ratio_arr = np.divide(erf_arr_a,normal_arr)

        #print("erf_arr_a=",erf_arr_a)
        #print("normal_arr=",normal_arr)
        #print("ratio_arr=",ratio_arr)
        ratio_min = min(ratio_arr)
        ratio_arg = np.argmin(ratio_arr)
        if ratio_init is None:
            ratio_init = ratio_min

        if not test_run:
            print("b=",b,"x12=",x1,x2,"ratio=",ratio_min,"est ratio=",ratio_min/(ratio_init*math.exp(-b*b)),"ratio a=",xlist[ratio_arg])

        normal_list_scaled = ratio_min*normal_list
        plt.plot(xlist, erf_list_a, lw = 1.0, label="erf2")
        plt.plot(xlist, normal_list_scaled, lw = 1.0, label="normal")
        plt.legend()
        if not test_run:
            plt.show()

    if test_run:
        print("compare_erf_normal OK")

if __name__ == "__main__":
    main(False) # test_run
