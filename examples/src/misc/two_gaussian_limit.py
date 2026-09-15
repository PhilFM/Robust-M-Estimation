import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import os
from pathlib import Path

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

def Fval(x:float, t:float, sigma:float, alpha:float):
    #print("x=",x,"t=",t,"sigma=",sigma,"alpha=",alpha)
    sigma_sqr = sigma*sigma
    exp1 = math.exp(-0.5*x*x)
    xt2 = (x-t)*(x-t)
    exp2 = math.exp(-0.5*xt2/sigma_sqr)
    return exp1 + alpha*exp2

def Fderiv(x:float, t:float, sigma:float, alpha:float):
    sigma_sqr = sigma*sigma
    exp1 = math.exp(-0.5*x*x)
    xt2 = (x-t)*(x-t)
    exp2 = math.exp(-0.5*xt2/sigma_sqr)
    return -x*exp1 - alpha*(x-t)*exp2/sigma_sqr

def F2ndderiv(x:float, t:float, sigma:float, alpha:float):
    sigma_sqr = sigma*sigma
    exp1 = math.exp(-0.5*x*x)
    xt2 = (x-t)*(x-t)
    exp2 = math.exp(-0.5*xt2/sigma_sqr)
    return (x*x - 1.0)*exp1 + alpha*(xt2/sigma_sqr - 1.0)*exp2/sigma_sqr

def F3rdderiv(x:float, t:float, sigma:float, alpha:float):
    sigma_sqr = sigma*sigma
    exp1 = math.exp(-0.5*x*x)
    xt2 = (x-t)*(x-t)
    exp2 = math.exp(-0.5*xt2/sigma_sqr)
    return (3.0*x - x*x*x)*exp1 + alpha*(3.0*(x-t)/sigma_sqr - xt2*(x-t)/(sigma_sqr*sigma_sqr))*exp2/(sigma*sigma)

def F_deriv_func(params: np.ndarray, # x, t, alpha
                 sigma:float):
    x = params[0]
    t = params[1]
    alpha = params[2]
    sigma_sqr = sigma*sigma
    exp1 = math.exp(-0.5*x*x)
    xt2 = (x-t)*(x-t)
    exp2 = math.exp(-0.5*xt2/sigma_sqr)
    deriv1 = -x*exp1 - alpha*(x-t)*exp2/sigma_sqr
    deriv2 = (x*x - 1.0)*exp1 + alpha*(xt2/sigma_sqr - 1.0)*exp2/sigma_sqr
    deriv3 = (3.0*x - x*x*x)*exp1 + alpha*(3.0*(x-t)/sigma_sqr - xt2*(x-t)/(sigma_sqr*sigma_sqr))*exp2/(sigma*sigma)
    return 1000.0*(deriv1*deriv1 + deriv2*deriv2 + deriv3*deriv3)

def F_func_x(params: np.ndarray, # x
             t:float,
             sigma:float,
             alpha:float):
    x = params[0]
    sigma_sqr = sigma*sigma
    exp1 = math.exp(-0.5*x*x)
    xt2 = (x-t)*(x-t)
    exp2 = math.exp(-0.5*xt2/sigma_sqr)
    return -100.0*(exp1 + alpha*exp2)

def test_sigma(sigma:float, test_run:bool, output_folder:str):
    sigma_sqr = sigma*sigma
    x = math.sqrt(sigma_sqr - 2.0 + 2.0*math.sqrt(sigma_sqr*sigma_sqr - sigma_sqr + 1.0))/sigma
    D2 = sigma_sqr*x*x + 3.0 - 3.0*sigma_sqr
    if D2 >= 0.0:
        t = x + sigma*math.sqrt(D2) # negative sign gives negative alpha
        alpha = -math.exp(-0.5*x*x)*math.exp(0.5*(x-t)*(x-t)/sigma_sqr)*sigma_sqr*x/(x-t)
        aterm1 = math.exp(0.5*(sigma_sqr - 1.0)*(x*x - 3.0))
        aterm2 = sigma*x/math.sqrt(D2)
        alpha2 = aterm1*aterm2
        if not test_run:
            print("OK: sigma=",sigma,"x=",x,"t=",t,"alpha=",alpha,alpha2,"D2=",D2)
            print("Alpha terms",aterm1,aterm2)

        # check 1st/2nd zero derivative formula
        D2p = sigma_sqr*x*x*x*x + sigma_sqr - 2.0*sigma_sqr*x*x + 4.0*x*x
        if not test_run:
            print("check zero:",x*t*t + (sigma_sqr*x*x - sigma_sqr - 2.0*x*x)*t + (1.0 - sigma_sqr)*x*x*x)
            print("check t",t," is one of",(sigma_sqr+2.0*x*x-sigma_sqr*x*x+sigma*math.sqrt(D2p))/(2.0*x),(sigma_sqr+2.0*x*x-sigma_sqr*x*x-sigma*math.sqrt(D2p))/(2.0*x))

        exp1 = math.exp(-0.5*x*x)
        xt2 = (x-t)*(x-t)
        exp2 = math.exp(-0.5*xt2/sigma_sqr)
        if not test_run:
            print("check equal",exp1/exp2,"to",math.exp((x*x - 2.0*x*t + t*t - sigma_sqr*x*x)/(2.0*sigma_sqr)))

        fid1 = sigma_sqr*sigma_sqr*x*x*x*x + sigma_sqr*sigma_sqr - 2.0*sigma_sqr*sigma_sqr*x*x + 2.0*sigma_sqr*x*x - 2.0*sigma_sqr*x*x*x*x
        fid2 = sigma_sqr*sigma*math.sqrt(D2p)*(x*x - 1.0)
        if not test_run:
            print("check equal(2)",exp1/exp2,"to either",math.exp((fid1 + fid2)/(4.0*sigma_sqr*x*x)),"or",math.exp((fid1 - fid2)/(4.0*sigma_sqr*x*x)))

        fid1 = (x-t)/x
        fid2 = ((x-t)*(x-t)/sigma_sqr - 1.0)/(x*x - 1.0)
        fid3 = (3.0*(x-t)/sigma_sqr - (x-t)*(x-t)*(x-t)/(sigma_sqr*sigma_sqr))/(3.0*x - x*x*x)
        if not test_run:
            print("fid123=",fid1,fid2,fid3)
            print("Equal?",sigma_sqr*x*x*x - sigma_sqr*t*x*x + sigma_sqr*t, x*x*x - 2.0*x*x*t + x*t*t)
            print("Equal2?",t-x,sigma*math.sqrt(sigma_sqr*x*x + 3.0 - 3.0*sigma_sqr))
            print("Zero?",(2.0*x*x - x*x*x*x + 3.0)*sigma_sqr*sigma_sqr + (x*x*x*x - 6.0*x*x - 3.0)*sigma_sqr + 4.0*x*x)

        val = Fval(x,t,sigma,alpha)
        if not test_run:
            print("Fval=",val,"Fderiv=",Fderiv(x,t,sigma,alpha),"F2ndderiv=",F2ndderiv(x,t,sigma,alpha),"F3rdderiv=",F3rdderiv(x,t,sigma,alpha))

        # check that we have a maximum
        if sigma <= 1.0:
            for tp in [t,t+2.0]: #np.linspace(t+0.1, t+5.0, num=20):
                for alphap in [alpha2,alpha2+0.03]:
                    xp = 0.0
                    xt2 = (xp-tp)*(xp-tp)
                    if not test_run:
                        print("  Second contribution:",math.exp(-0.5*xt2/sigma_sqr))

                    ok = True
                    try:
                        x1 = scipy.optimize.newton(Fderiv, x0=0.0, args=(tp,sigma,alphap))
                    except RuntimeError:
                        if not test_run:
                            print("  Newton failed from x=0")

                        ok = False

                    try:
                        x2 = scipy.optimize.newton(Fderiv, x0=tp, args=(tp,sigma,alphap))
                    except RuntimeError:
                        if not test_run:
                            print("  Newton failed from x=",tp)

                        ok = False

                    if ok:
                        fval1 = Fval(x1,tp,sigma,alphap)
                        fval2 = Fval(x2,tp,sigma,alphap)
                        fderiv1 = Fderiv(x1,tp,sigma,alphap)
                        fderiv2 = Fderiv(x2,tp,sigma,alphap)
                        if not test_run:
                            print("vals",fval1,fval2,"derivs",fderiv1,fderiv2)
                            if abs(x2-x1) < 0.0001:
                                print("  No separation for tp=",tp,"x12=",x1,x2)
                            elif fval2 > fval1:
                                print("  problem: x=",x1,x2,"tp=",tp,"Fval_diff=",fval2-fval1)
                            else:
                                print("  peak OK: x=",x1,x2,"tp=",tp,"Fval_diff=",fval2-fval1)
                    else:
                        if not test_run:
                            print("  Fail")

                    if True:
                        # plot function
                        xlist = np.linspace(0.0, 6.0, 400)
                        plt.close("all")
                        plt.figure(num=1, dpi=240)
                        funcX = np.vectorize(Fval, excluded={"t","sigma","alpha"})
                        plt.plot(xlist, funcX(xlist, t=tp, sigma=sigma, alpha=alphap), lw = 1.0, label="Separation")

                        #plt.legend()
                        plt.savefig(os.path.join(output_folder, "two_gaussian_limit.png"), bbox_inches='tight')
                        if not test_run:
                            plt.show()
            
    else:
        print("Error: sigma=",sigma,"x=",x,"D2=",D2)

    if False:
        res = scipy.optimize.minimize(F_deriv_func, x0=np.array([0.5*sigma,sigma,0.5]), args=(sigma))
        if res.success:
            x = res.x[0]
            t = res.x[1]
            alpha = res.x[2]
            if not test_run:
                print("Minimise worked func=",res.fun,": x=",x,",t=",t,",alpha=",alpha)

            fid1 = (x-t)/x
            fid2 = ((x-t)*(x-t)/sigma_sqr - 1.0)/(x*x - 1.0)
            fid3 = (3.0*(x-t)/sigma_sqr - (x-t)*(x-t)*(x-t)/(sigma_sqr*sigma_sqr))/(3.0*x - x*x*x)
            if not test_run:
                print("Minimise: fid123=",fid1,fid2,fid3)
                print("Minimise: Equal?",sigma_sqr*x*x*x - sigma_sqr*t*x*x + sigma_sqr*t, x*x*x - 2.0*x*x*t + x*t*t)
                print("Minimise: Equal2?",t-x,sigma*math.sqrt(sigma_sqr*x*x + 3.0 - 3.0*sigma_sqr))
                print("Minimise: Zero?",(2.0*x*x - x*x*x*x + 3.0)*sigma_sqr*sigma_sqr + (x*x*x*x - 6.0*x*x - 3.0)*sigma_sqr + 4.0*x*x)
                print("Minimise: Fval=",Fval(x,t,sigma,alpha),"Fderiv=",Fderiv(x,t,sigma,alpha),"F2ndderiv=",F2ndderiv(x,t,sigma,alpha),"F3rdderiv=",F3rdderiv(x,t,sigma,alpha))
        else:
            if not test_run:
                print("Minimise failed")

    if test_run:
        print("two_gaussian_limit OK")

def main(test_run:bool, output_folder:str="../../output"):
    output_folder += "/misc"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for sigma in [0.5]: #np.linspace(0.1, 2.0, num=20):
        test_sigma(sigma, test_run, output_folder)

if __name__ == "__main__":
    main(False) # test_run
