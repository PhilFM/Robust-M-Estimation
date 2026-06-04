import math
import numpy as np
import os
import sys
import scipy
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sys.path.append("../../../pypi_package/src")

from gnc_smoothie.sup_gauss_newton import SupGaussNewton
from gnc_smoothie.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie.linear_model.linear_regressor import LinearRegressor
from gnc_smoothie.linear_model.linear_regressor_welsch import LinearRegressorWelsch
from gnc_smoothie.cython_files.linear_regressor_welsch_evaluator import LinearRegressorWelschEvaluator

sys.path.append("../misc")
from minimiser import minimiser

def randomM11() -> float:
    return 2.0*(np.random.rand()-0.5)

# check linear approximation w.r.t. c for function erf(c*x + d)
def check_erf_approx(d: float,
                     x: float):
    for c in np.linspace(0.0, 5.0, num=5000):
        print("c:",c,"erf:",math.erf(c*x+d),"approx ratio:",(math.erf(d) + 2.0*c*math.exp(-d ** 2)*x/math.pi)/math.erf(c*x+d))

def line_segment_integral_small_c(d: float,
                                  x1: float,
                                  x2: float):
    return (x2 - x1)*math.exp(-d ** 2)

def line_segment_integral_large_c(c: float,
                                  d: float,
                                  x1: float,
                                  x2: float):
    return 0.5*math.sqrt(math.pi)*(math.erf(c*x2 + d) - math.erf(c*x1 + d))/c

def line_segment_integral(c: float,
                          d: float,
                          x1: float,
                          x2: float):
    if abs(c) < 0.00001:
        return (x2 - x1)*math.exp(-d ** 2) # line_segment_integral_small_c(d, x1, x2)
    else:
        return 0.5*math.sqrt(math.pi)*(math.erf(c*x2 + d) - math.erf(c*x1 + d))/c # line_segment_integral_large_c(c, d, x1, x2)

def line_segment_integral_linear_c(c: float,
                                   d: float,
                                   x1: float,
                                   x2: float):
    return math.exp(-d ** 2)*(x2 - x1)*(1.0 - 4.0*c*d*(x1 + x2)/math.pi)

def check_line_segment_func(d: float,
                            x1: float,
                            x2: float):
    print("")
    print("d=",d,"x1=",x1,"x2=",x2)
    for c in np.linspace(0.001, 5.0, num=5000):
        print("c=",c,"large:",line_segment_integral_large_c(c,d,x1,x2),"small:",line_segment_integral_small_c(d,x1,x2),"linear:",line_segment_integral_linear_c(c,d,x1,x2),"ratio:",line_segment_integral_linear_c(c,d,x1,x2)/line_segment_integral(c,d,x1,x2))

sigma = 0.1
sig_sqrt_2 = sigma*math.sqrt(2.0)
inv_sig_sqrt_2 = 1.0/sig_sqrt_2
x_half_range = 0.5

def line_segment_func_neg(ab: np.ndarray,
                          ap: float,
                          bp: float,
                          alpha: float):
    if alpha > 0.0:
        c = (ab[0] - ap)*inv_sig_sqrt_2
        d = (ab[1] - bp)*inv_sig_sqrt_2
        x1 = -x_half_range
        x2 = x_half_range*(2.0*alpha - 1.0)
        if abs(c) < 0.00001:
            v1 = (x2 - x1)*math.exp(-d ** 2) # line_segment_integral_small_c(d, x1, x2)
            #print("v1=",v1,"x12=",x1,x2,"alpha=",alpha,"abp=",ap,bp,"d=",d,"expd=",math.exp(-d ** 2))
        else:
            v1 = 0.5*math.sqrt(math.pi)*(math.erf(c*x2 + d) - math.erf(c*x1 + d))/c # line_segment_integral_large_c(c, d, x1, x2)
    else:
        v1 = 0.0

    if alpha < 1.0:
        c = ab[0]*inv_sig_sqrt_2
        d = ab[1]*inv_sig_sqrt_2
        x1 = x_half_range*(2.0*alpha - 1.0)
        x2 = x_half_range
        if abs(c) < 0.00001:
            v2 = (x2 - x1)*math.exp(-d ** 2) # line_segment_integral_small_c(d, x1, x2)
        else:
            v2 = 0.5*math.sqrt(math.pi)*(math.erf(c*x2 + d) - math.erf(c*x1 + d))/c # line_segment_integral_large_c(c, d, x1, x2)
    else:
        v2 = 0.0

    return -v1 - v2

def line_segment_func_p(a: float,
                        b: float,
                        ap: float,
                        bp: float,
                        alpha: float):
    if alpha > 0.0:
        v1 = line_segment_integral((a-ap)*inv_sig_sqrt_2, (b-bp)*inv_sig_sqrt_2, -x_half_range, x_half_range*(2.0*alpha-1.0))
    else:
        v1 = 0.0

    if alpha < 1.0:
        v2 = line_segment_integral(a*inv_sig_sqrt_2, b*inv_sig_sqrt_2, x_half_range*(2.0*alpha-1.0), x_half_range)
    else:
        v2 =  0.0

    return  v1+v2

def show_ab_func(ap: float,
                 bp: float,
                 alpha: float):
    a_min = min(-3.0, 2.0*ap)
    a_max = max( 3.0, 2.0*ap)
    alist = np.linspace(a_min, a_max, 200)
    hmfv = np.vectorize(line_segment_func_p, excluded={"b","ap","bp","alpha"})
    plt.close("all")
    plt.figure(num=1, dpi=240)
    plt.axline((a_min, 0.0), (a_max, 0.0), color = (0.6,0.6,0.6), linewidth=0.5)
    nb = 11
    best_af = None
    for i,b in enumerate(np.linspace(min(-3.0, 3.0*bp), max(3.0, 3.0*bp), nb)):
        flist = hmfv(alist, b=b, ap=ap, bp=bp, alpha=alpha)
        a_idx = np.argmax(flist)
        if best_af is None or flist[a_idx] > best_af[1]:
            best_af = (alist[a_idx], flist[a_idx])

        plt.plot(alist, flist, lw = 1.0, label="b="+str(b))

    print("best_af=",best_af)
    out_vp = (ap, -line_segment_func_neg(np.array([ap,bp]),ap,bp,alpha))
    plt.axline((out_vp[0], 0.0), out_vp, color = (1.0,0.0,0.0), linewidth=0.5)

    res1 = scipy.optimize.minimize(line_segment_func_neg, x0=np.array([ap,bp]), args=(ap,bp,alpha))
    res2 = scipy.optimize.minimize(line_segment_func_neg, x0=np.array([0.0,0.0]), args=(ap,bp,alpha))
    res = None
    if res1.success and res2.success:
        if res1.fun < res2.fun:
            res = res1
        else:
            res = res2
    elif res1.success:
        res = res1
    elif res2.success:
        res = res2

    if res.success:
        # check that we actually have a minimum
        print("abp=",ap,bp,"alpha=",alpha,"res.x=",res.x,"v=",-res.fun,"vp=",-line_segment_func_neg(np.array([ap,bp]),ap,bp,alpha),-line_segment_func_neg(np.array([0.0,0.0]),ap,bp,alpha))
        plt.axline((res.x[0], 0.0), (res.x[0],-res.fun), color = (0.0,0.0,1.0), linewidth=1.0, label="Max: b="+str(res.x[1]))

    plt.legend()
    plt.show()

# show variation w.r.t. a in line integral function
#   F(a,b) = f(a-a',b-b',-1,-1+alpha) + f(a,b,-1+alpha,1)
def check_breakdown():
    show_ab_func(-1.0, 0.7, 0.2) # ap, bp, alpha
    show_ab_func(-1.0, 0.7, 0.8) # ap, bp, alpha
    show_ab_func(-4.0, 3.0, 0.2) # ap, bp, alpha
    show_ab_func(-4.0, 3.0, 0.5) # ap, bp, alpha
    show_ab_func(-4.0, 3.0, 0.8) # ap, bp, alpha
    show_ab_func(-4.0, 3.0, 1.0) # ap, bp, alpha

def show_c_func(d: float,
                x1: float,
                x2: float):
    clist = np.linspace(-10.0, 10.0, 200)
    hmfv = np.vectorize(line_segment_integral, excluded={"d","x1","x2"})
    lmfv = np.vectorize(line_segment_integral_linear_c, excluded={"d","x1","x2"})
    plt.close("all")
    plt.figure(num=1, dpi=120)
    plt.plot(clist, hmfv(clist, d=d, x1=x1, x2=x2), lw = 1.0)
    plt.plot(clist, lmfv(clist, d=d, x1=x1, x2=x2), lw = 1.0)
    plt.show()

# show linear approximation to f(c,d,x1,x2) = sqrt(pi)/(2*c)*(erf(c*x2+d) - erf(c*x1+d))
def check_breakdown2():
    for d in np.linspace(0.0, 10.0, 101):
        show_c_func(d, -1.0, 1.0)

def check_breakdown3():
    anum = 11
    bnum = 21
    alpha_list = np.linspace(0.0,0.5, num=101)
    alist = np.linspace(0.0,10.0,num=anum)
    blist = np.linspace(-10.0,10.0,num=bnum)

    small_diff = 1.e-3
    for alpha in alpha_list:
        print("alpha=",alpha)
        for api,ap in enumerate(alist):
            for bpi,bp in enumerate(blist):
                absol = []
                for a in [-1.0,0.0,1.0,ap]:
                    for b in [-20.0,0.0,20.0,bp]:
                        res = scipy.optimize.minimize(line_segment_func_neg, x0=np.array([a,b]), args=(ap,bp,alpha))
                        if res.success:
                            # check that we actually have a minimum
                            v = line_segment_func_neg(res.x,ap,bp,alpha)
                            van = line_segment_func_neg([res.x[0]-small_diff,res.x[1]],ap,bp,alpha)
                            vap = line_segment_func_neg([res.x[0]+small_diff,res.x[1]],ap,bp,alpha)
                            vbn = line_segment_func_neg([res.x[0],res.x[1]-small_diff],ap,bp,alpha)
                            vbp = line_segment_func_neg([res.x[0],res.x[1]+small_diff],ap,bp,alpha)
                            if v < van and v < vap and v < vbn and v < vbp:
                                found = False
                                for ab in absol:
                                    if np.linalg.norm(res.x-ab) < 1.e-3:
                                        found = True

                                if not found:
                                    absol.append(res.x)
                            else:
                                pass #print("v=",v,"other v:",van-v,vap-v,vbn-v,vbp-v)
                        else:
                            print("message=",res.message)

                print("  abp=",ap,bp,"sol=",absol)

                #func_ab = np.zeros((anum,bnum))
                #for ai,a in enumerate(alist):
                #    for bi,b in enumerate(blist):
                #        func_ab[ai][bi] =   line_segment_integral((a - ap)*inv_sig_sqrt_2, (b - ap)*inv_sig_sqrt_2, -0.5, -0.5+alpha) + line_segment_integral(a*inv_sig_sqrt_2, b*inv_sig_sqrt_2, -0.5+alpha, 0.5)
                
def check_breakdown4():
    sigma = 0.5
    sigma_base = 0.01
    D = 20.0
    n_points = 500
    for n_bad_points in range(1,n_points//2):
        data = np.zeros((n_points,2))
        n_good_points = n_points - n_bad_points

        for yscale in [1.0,1.5,2.0,2.5,3.0]:
            for alpha in [0.5,0.6,0.7,0.8,0.9]:
                # we slant the bad line to insersect the good data, i.e. we calculate the line intersecting the two points
                # x = -D + 0.5*n_bad_points*2.0*D/(n_points-1), y = yscale*sigma
                # x = -D + (n_bad_points+alpha*n_good_points)*2.0*D/(n_points-1), y = 0.0, where alpha = 0.75
                # y1 = a*x1 + b 
                # y2 = a*x2 + b gives a = (y2 - y1)/(x2 - x1), b = y1 - a*x1
                x1 = -D + 0.5*n_bad_points*2.0*D/(n_points-1)
                y1 = yscale*sigma
                x2 = -D + (n_bad_points+alpha*n_good_points)*2.0*D/(n_points-1)
                y2 = 0.0
                aout = (y2 - y1)/(x2 - x1)
                bout = y1 - aout*x1
                for i in range(n_bad_points):
                    x = -D + i*2.0*D/(n_points-1)
                    data[i][0] = x
                    data[i][1] = aout*x + bout

                for i in range(n_bad_points,n_points):
                    x = -D + i*2.0*D/(n_points-1)
                    data[i][0] = x
                    data[i][1] = 0.0

                detA_list = []
                optc_list = []
                for sig in [0.01,0.05,0.1,0.2,0.5,0.8,2.0,5.0]:
                    param_instance = GNC_WelschParams(WelschInfluenceFunc(), sigma_base=sig)
                    evaluator_instance = LinearRegressorWelschEvaluator(data[0])
                    optimiser_instance = SupGaussNewton(param_instance, data, evaluator_instance=evaluator_instance)
                    a, A = optimiser_instance.weighted_derivs(np.array([0.0,0.0]), 1.0) # lambda_b
                    detA = A[0,0]*A[1,1]-A[0,1]*A[1,0]
                    detA_list.append(float(detA))
                    optc_list = optimiser_instance.objective_func(np.array([0.0,0.0])) - optimiser_instance.objective_func(np.array([aout,bout]))

                x_range = max(data[:,0]) - min(data[:,0])
                y_range = max(data[:,1]) - min(data[:,1])
                line_fitter = LinearRegressorWelsch(sigma_base=sigma_base, sigma_limit=y_range, num_sigma_steps=50, debug=True, max_niterations=200)
                if line_fitter.run(data):
                    debug_line_list = line_fitter.debug_model_list
                    # "ab=",a,b,
                    aratio = float(line_fitter.final_model[0])/aout
                    print("Outlier ratio ",n_bad_points/n_points,"yscale=",yscale,"alpha=",alpha,"2nd deriv. det:", detA_list,"Line parameters:",line_fitter.final_model,"aratio=",aratio)
                    if aratio > 0.9:
                        print("   line list:",debug_line_list)
                
def check_breakdown5():
    sigma = 0.1
    D = 20.0
    n_points = 500
    for n_bad_points in range(1,n_points//2):
        data = np.zeros((n_points,2))
        for i in range(n_bad_points):
            x = -D + i*2.0*D/(n_points-1)
            data[i][0] = x
            data[i][1] = 2.0*sigma

        # n_good_points = n_points - n_bad_points
        for i in range(n_bad_points,n_points):
            x = -D + i*2.0*D/(n_points-1)
            data[i][0] = x
            data[i][1] = 0.0

        param_instance = GNC_WelschParams(WelschInfluenceFunc(), sigma_base=sigma)
        evaluator_instance = LinearRegressorWelschEvaluator(data[0])
        optimiser_instance = SupGaussNewton(param_instance, data, evaluator_instance=evaluator_instance)
        def objective_func(x: np.array) -> float:
            # The line should intersect the point x = -D + (n_bad_points/2)*2.0*D/(n_points-1), y = 2*sigma
            # So given a in y = a*x + b, we have b = y - a*x
            a = x[0]
            xp = -D + (n_bad_points/2)*2.0*D/(n_points-1)
            y = 2.0*sigma
            b = y - a*xp
            return -optimiser_instance.objective_func(np.array([a,b]))

        ab_max,best_val = minimiser(objective_func, initial_centre=[0.0], initial_half_range=[2.0], n_samples=[41], scale_factor=1.4)
        good_val = optimiser_instance.objective_func(np.array([0.0,0.0]))
        print("Compare (",n_bad_points/n_points,")",-best_val,good_val,-best_val-good_val)
        
def main(test_run:bool, output_folder:str="../../../output", quick_run:bool=False):
    np.random.seed(0) # We want the numbers to be the same on each run

    #check_erf_approx(2.0, -1.0)
    #check_line_segment_func(1.3, -0.3, 0.1)
    #check_line_segment_func(-3.0, 0.5, 1.5)
    check_breakdown()
    check_breakdown2()
    check_breakdown3()
    check_breakdown4()
    check_breakdown5()

    outlier_ratio = 0.5-0.0001
    n_points = 20 if quick_run else 100
    n_bad_points = int(outlier_ratio*n_points)
    #print("n_bad_points=",n_bad_points)
    sigma_pop = 0.3
    q = 0.666667
    sigma_base = sigma_pop/q
    sigma_limit = 5.0
    num_sigma_steps = 3 if quick_run else 20

    line_good = [0.0, 0.0]

    D = 3.0
    data = np.zeros((n_points,2))
    for i in range(n_points):
        x = -D + i*2.0*D/(n_points-1)
        data[i][0] = x
        data[i][1] = 0.0 #np.random.normal(0.0,sigma_pop) # good line is a=b=0

    influence_func = WelschInfluenceFunc()
    evaluator_instance = LinearRegressorWelschEvaluator(data[0])
    #model_instance = LinearRegressor(data[0])
    param_instance = GNC_WelschParams(influence_func, sigma_base,
                                      sigma_limit=sigma_limit, num_sigma_steps=num_sigma_steps)

    if True:
        optimiser_instance_good = SupGaussNewton(param_instance, data, evaluator_instance=evaluator_instance)
        optimiser_instance_good._param_instance.reset(init=False)
        def objective_func(x: np.array) -> float:
            #print("x=",x)
            a, A = optimiser_instance_good.weighted_derivs(np.array(x), 1.0) # lambda_b
            return np.sum(A**2)

        ab_max,best_val = minimiser(objective_func, initial_centre=[0.0, 0.0], initial_half_range=[2.0, 1.0], n_samples=[41,41], scale_factor=1.4)
        a, A = optimiser_instance_good.weighted_derivs(ab_max, 1.0) # lambda_b
        print("Curvature minimiser ab_max",ab_max,best_val,A)

        n_a_samples = 40
        a_half_range = 4.0*sigma_base/D
        for a_idx in range(n_a_samples):
            av = -a_half_range + 2.0*a_half_range*a_idx/(n_a_samples-1)
            print("a=",av,"f=",optimiser_instance_good.objective_func(np.array([av,0.0])),"thres=",math.exp(-0.5)*n_points,2.0*sigma_base/D)

    bad_point_scale = 5.0
    for test_idx in range(1000):
        data_c = np.copy(data)
        bad_point = 0.002*test_idx
        for i in range(n_bad_points):
            data_c[i][1] = bad_point

        x_offset = 0.0
        for i in range(n_bad_points,n_points):
            x_offset += data[i][0]

        x_offset /= n_points-n_bad_points

        # calculate curvature at ground truth
        optimiser_instance = SupGaussNewton(param_instance, data_c, evaluator_instance=evaluator_instance)
        optimiser_instance._param_instance.reset(init=False)

        if True:
            def objective_func(x: np.array) -> float:
                #print("x=",x)
                return -optimiser_instance.objective_func(x)

            #print("")
            ab_max,best_val = minimiser(objective_func, initial_centre=[0.0, 0.0], initial_half_range=[4.0, 5.0], n_samples=[21,21], scale_factor=2.0)

            #print("bad_point:",bad_point,"ab_max",ab_max,best_val)
            ab_max[1] += x_offset*ab_max[0] # centre on good data
            ab_max[0] /= D
            if abs(test_idx-450) < 3 or abs(ab_max[1]) > sigma_base: # or abs(ab_max[0]) > sigma_base
                print("vmax=",-best_val,"ref=",optimiser_instance.objective_func(np.array([0.0,0.0])),optimiser_instance.objective_func(np.array([0.0,sigma_base])),"bad point ",bad_point,ab_max,"normalised max distance:",ab_max[0]/(D*sigma_base), ab_max[1]/sigma_base)

        if False:
            a, A = optimiser_instance.weighted_derivs(np.array(line_good), 1.0) # lambda_b
            Asum = np.zeros((2,2))
            inv_sigma4 = math.pow(sigma_base, -4.0)
            #print("test_idx=",test_idx)
            for d in data_c:
                r = d[1] - line_good[0]*d[0] - line_good[1] # residual
                #print("r=",r)
                H = np.array([d[0], 1.0])
                HTH = np.outer(H,H)
                Asum += inv_sigma4*math.exp(-0.5*r*r/(sigma_base*sigma_base))*(r*r - sigma_base*sigma_base)*HTH

            detA = Asum[0][0]*Asum[1][1]-Asum[0][1]*Asum[1][0]
            if detA <= 0.0:
                print("A=",A,"Asum=",Asum)
                #print("data_c=",data_c)
                print("bad_point=",bad_point,"det=",detA)

            assert(detA > 0.0)

    if test_run:
        print("line_fit_breakdown2 OK")

if __name__ == "__main__":
    main(False) # test_run
