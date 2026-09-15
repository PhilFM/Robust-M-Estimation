import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
from pathlib import Path

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

def diff_func_deriv(t: float, x:float):
    #print("t=",t,"x=",x)
    return (x-t)*math.exp(-0.5*(x-t)*(x-t)) + t*math.exp(-0.5*t*t)

def diff_max_func(x:float):
    #print("x=",x)
    try:
        t = scipy.optimize.newton(diff_func_deriv, x0=x, args=[x])
        return t
    except RuntimeError:
        print("Failed x=",x)
        return 0.0

def main(test_run:bool, output_folder:str="../../output"):
    output_folder += "/misc"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for x in (0.5,1.0,2.0,10.0):
        diff_max_val = diff_max_func(x)
        if not test_run:
            print("diff_max_func(",x,")=",diff_max_val)

    x_max = 5.0
    xlist = np.linspace(0.4, x_max, 100)
    tlist = np.zeros(len(xlist))
    for i,x in enumerate(xlist):
        tlist[i] = diff_max_func(x)

    plt.close("all")
    plt.figure(num=1, dpi=240)
    ax = plt.gca()
    #plt.box(False)
    #ax.set_xlim((x_min, x_max))
    ax.set_ylim((0.0, x_max))
    plt.plot(xlist, tlist, lw = 1.0, label="Diff max")

    #plt.legend()
    #plt.savefig(os.path.join(output_folder, "difference_max.png"), bbox_inches='tight')
    if not test_run:
        plt.show()
    
    if test_run:
        print("difference_max OK")

if __name__ == "__main__":
    main(False) # test_run
