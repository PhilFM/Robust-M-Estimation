import numpy as np
import math
import matplotlib.pyplot as plt

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

def func(x, xp, inv_var):
    return 1.0 - math.exp(-0.5*(x - xp)*(x - xp)*inv_var) - math.exp(-0.5*x*x*inv_var) + math.exp(-0.5*xp*xp*inv_var)

def main(test_run:bool, output_folder:str="../../output"):
    sigma = 1.0
    inv_var = 1.0/(sigma*sigma)
    xlist = np.linspace(0.1, 3.0, num=30)
    xplist = np.linspace(-5.0, 5.0, num=1001)
    small_val = 0.0 # 1.e-10
    for x in xlist:
        print("x = ",x)
        last_val = None
        for xp in xplist:
            fv = func(x, xp, inv_var)
            if last_val is not None:
                if fv < -small_val and last_val >= 0.0:
                    print("   Switching to negative at xp=",xp)
                elif fv > small_val and last_val <= 0.0:
                    print("   Switching to positive at xp=",xp)

            #print("   xp=",xp,"fv=",fv)
            last_val = fv

if __name__ == "__main__":
    main(False) # test_run
