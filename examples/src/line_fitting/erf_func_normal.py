import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
from pathlib import Path

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

sqrt_2 = math.sqrt(2.0)
sqrt_pi = math.sqrt(math.pi)

def show_cross_section():
    N = 10000
    sigma_l = 1.0 #4.0
    x_half_range = 8.0*sigma_l # covers the range of non-vanishing vallues
    (ab,bb) = (1.0,2.0)
    alpha = 0.3
    x0 = sigma_l*scipy.special.erfinv(2.0*alpha-1.0) # so alpha = 0.5*(1+erf(x0/sigma_l))
    sigma = 2.0
    sup_gn_q = 0.3 #0.6667
    sigma_pop = sup_gn_q*sigma
    inv_var = 1.0/(sigma*sigma)
    half_inv_var = 0.5*inv_var
    sigma_c = math.sqrt(sigma*sigma + sigma_pop*sigma_pop)
    inv_var_c = 1.0/(sigma_c*sigma_c)
    half_inv_var_c = 0.5*inv_var_c
    (ab_canon,bb_canon) = (sigma_l*ab/sigma, bb/sigma)
    data_x = np.linspace(-x_half_range, x_half_range, N)
    data_y = np.zeros(N)
    weight = np.zeros(N)
    inv_var_l = 1.0/(sigma_l*sigma_l)
    half_inv_var_l = 0.5*inv_var_l
    n_bad_points = int(0.5*N*(x_half_range+x0)/x_half_range)
    totw = 0.0
    for i in range(N):
        weight[i] = math.exp(-half_inv_var_l*(data_x[i] ** 2.0))
        totw += weight[i]

    # rescale weights so that they add up to 1
    weight /= totw

    # build y values - first bad points
    for i in range(n_bad_points):
        data_y[i] = ab*data_x[i] + bb

    # now good points
    for i in range(n_bad_points,N):
        data_y[i] = np.random.normal(0.0,sigma_pop)

    xlist = np.linspace(-2.0, 2.0, 201)
    list_bad = np.zeros(len(xlist))
    list_bad_canon = np.zeros(len(xlist))
    list_bad_data = np.zeros(len(xlist))
    list_good = np.zeros(len(xlist))
    list_good_canon = np.zeros(len(xlist))
    list_good_data = np.zeros(len(xlist))

    norm = math.sqrt(ab*ab + bb*bb)
    acoef = ab/norm
    bcoef = bb/norm
    F_fac_sqr = 1.0/(1.0 + sup_gn_q**2)
    F_fac = math.sqrt(F_fac_sqr)
    for i,x in enumerate(xlist):
        def F_good(a:float, b:float):
            u = math.sqrt(inv_var_c*a*a + inv_var_l)/sqrt_2
            v = 0.5*inv_var_c*a*b/u
            w = half_inv_var_c*b*b - v*v
            return (sigma/sigma_c)*math.exp(-w)*(1.0 - math.erf(u*x0+v))/u

        def F_good_canon(a:float, b:float): # assumes sigma = sigma_l = 1
            u = math.sqrt(F_fac_sqr*a*a + 1.0)/sqrt_2
            v = 0.5*F_fac_sqr*a*b/u
            w = 0.5*F_fac_sqr*b*b - v*v
            return F_fac*math.exp(-w)*(1.0 - math.erf(u*x0+v))/u

        def F_good_data(a:float, b:float):
            tot = 0.0
            for i in range(n_bad_points,N):
                tot += weight[i]*math.exp(-half_inv_var*((data_y[i]-a*data_x[i]-b) ** 2.0))

            return sigma_l*tot*2.0*sqrt_2

        def F_bad(a:float, b:float):
            ad = ab-a
            bd = bb-b
            u = math.sqrt(inv_var*ad*ad + inv_var_l)/sqrt_2
            v = 0.5*inv_var*ad*bd/u
            w = half_inv_var*bd*bd - v*v
            return math.exp(-w)*(1.0 + math.erf(u*x0+v))/u

        def F_bad_canon(a:float, b:float): # assumes sigma = sigma_l = 1
            ad = ab_canon-a
            bd = bb_canon-b
            u = math.sqrt(ad*ad + 1.0)/sqrt_2
            v = 0.5*ad*bd/u
            w = 0.5*bd*bd - v*v
            return math.exp(-w)*(1.0 + math.erf(u*x0+v))/u

        def F_bad_data(a:float, b:float):
            tot = 0.0
            for i in range(n_bad_points):
                tot += weight[i]*math.exp(-half_inv_var*((data_y[i]-a*data_x[i]-b) ** 2.0))

            return sigma_l*tot*2.0*sqrt_2

        a = x*acoef
        b = x*bcoef
        list_bad[i] = F_bad(a,b)
        list_bad_canon[i] = F_bad_canon(a*sigma_l/sigma,b/sigma)
        list_bad_data[i] = F_bad_data(a, b)

        list_good[i] = F_good(a,b)
        list_good_canon[i] = F_good_canon(a*sigma_l/sigma,b/sigma)
        list_good_data[i] = F_good_data(a, b)
        #print("x=",x,"ratios=",list_bad[i]/list_bad_data[i],list_good[i]/list_good_data[i])

    plt.close("all")
    plt.figure(num=1, dpi=240)
    #ax = plt.gca()
    #plt.box(False)
    #ax.set_xlim((x_min, x_max))
    #ax.set_ylim((0.0, 1.2))
    # a = x*a/norm, b = x*b/norm, so x = norm
    #abx = norm
    #plt.axline((0.0, 0.0), (0.0, 0.5), color = "yellow", linewidth=1)
    #plt.axline((abx, 0.0), (abx, 0.5), color = "magenta", linewidth=1)
    plt.plot(xlist, list_bad, lw = 1.0, label="F bad", color = "red")
    plt.plot(xlist, list_bad_canon, lw = 1.5, label="F bad canon", color = "cyan", linestyle = "dotted")
    plt.plot(xlist, list_bad_data, lw = 1.5, label="F bad data", color = "green", linestyle = "dashed")
    plt.legend()
    plt.show()

    plt.close("all")
    plt.figure(num=1, dpi=240)
    #ax = plt.gca()
    #plt.box(False)
    #ax.set_xlim((x_min, x_max))
    #ax.set_ylim((0.0, 1.2))
    # a = x*a/norm, b = x*b/norm, so x = norm
    #abx = norm
    #plt.axline((0.0, 0.0), (0.0, 0.5), color = "yellow", linewidth=1)
    #plt.axline((abx, 0.0), (abx, 0.5), color = "magenta", linewidth=1)
    plt.plot(xlist, list_good, lw = 1.0, label="F good", color = "red")
    plt.plot(xlist, list_good_canon, lw = 1.5, label="F good canon", color = "cyan", linestyle = "dotted")
    plt.plot(xlist, list_good_data, lw = 1.5, label="F good data", color = "green", linestyle = "dashed")
    plt.legend()
    plt.show()
                
def main(test_run:bool, output_folder:str="../../output"):
    output_folder += "/line_fit"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    show_cross_section()

if __name__ == "__main__":
    main(False) # test_run
