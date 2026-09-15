# Compares Fc(u,v) = (erf(u) + erf(v)/(u+v) with Gaussian function
# F_good(a,b) = N'*F_fac*Fc(-F_fac*(a*D+b)/(sqrt(2)*sigma),F_fac*(a*xbl+b)/(sqrt(2)*sigma))
# F_bad(a,b) = N*Fc(-((a-ab)*D+bb-b)/(sqrt(2)*sigma),((ab-a)*xbl+bb-b)/(sqrt(2)*sigma))
# alpha = N/(N'+N)
# xbl = -D + 2*alpha*D
#
# where
# sigma is the Sup-GN scale
# q is the Sup-GN scale ratio
# sigma_pop is the estimated noise
# sigma_pop = q*sigma
# F_fac = 1.0/sqrt(1 + q^2)
# D = half-range of data
# N = number of outlier points
# N' = number of good points


import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

sqrt_2 = math.sqrt(2.0)
sqrt_pi = math.sqrt(math.pi)

F_canon_thres = 0.00001
def F_canon(u,v):
    upv = u+v
    if abs(upv) < F_canon_thres:
        return 2.0*math.exp(-u*u)/sqrt_pi
    else:
        return (math.erf(u) + math.erf(v))/upv

def normal_func(x,sigma,offset,scale):
    #print("x=",x,"sigma=",sigma,"offset=",offset,"scale=",scale,math.exp(-0.5*x*x/(sigma*sigma)))
    return offset + scale*math.exp(-0.5*x*x/(sigma*sigma))

def check_F_canon(a:float, test_run:bool, output_folder:str):
    val1 = F_canon(a,a+0.9*F_canon_thres)
    val2 = F_canon(a,a+1.1*F_canon_thres)
    if not test_run:
        print("a=",a,"check ",val1,val2,"diff=",val2-val1)

def visualise_F_canon(test_run:bool, output_folder:str):
    uv_range = 5.0
    imsize = 512
    imarr = np.zeros((imsize,imsize), dtype=np.uint8)
    for i,v in enumerate(np.linspace(-uv_range, uv_range, num=imsize)):
        for j,u in enumerate(np.linspace(-uv_range, uv_range, num=imsize)):
            imarr[imsize-i-1][j] = np.uint8(255.0*F_canon(u,v)/1.2)

    im = Image.fromarray(imarr)
    im.save("test.png")
    
def show_F_canon(test_run:bool, output_folder:str):
    w_range = 5.0
    wlist = np.linspace(-w_range, w_range, num=101)
    small_val = 2.0*w_range/(len(wlist)-1)
    normal_mfv = np.vectorize(normal_func, excluded={"sigma", "offset", "scale"})

    if not test_run:
        print("Compare ",1.1283791670955126,2.0/sqrt_pi)

    for theta in np.linspace(0.0, math.pi, 21):
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # model upper bound as offset + scaled Gaussian
        F_canon_sigma_high = math.sqrt(1.5*(costheta + sintheta)/(sintheta ** 3 + costheta ** 3))
        F_canon_offset_high = abs(costheta + sintheta)*0.36
        F_canon_scale_high = 2.0/sqrt_pi - F_canon_offset_high

        # model lower bound as scaled Gaussian
        F_canon_offset_low = 0.0
        F_canon_scale_low = 2.0/sqrt_pi
        F_canon_sigma_low = math.sqrt(1.5*(costheta + sintheta)/(sintheta ** 3 + costheta ** 3))

        Fclist = np.zeros(len(wlist))
        for i,w in enumerate(wlist):
            u = w*costheta
            v = w*sintheta
            Fclist[i] = F_canon(u,v)

        Fmax_arg = np.argmax(Fclist)
        if not test_run:
            print("theta=",theta,theta/math.pi,"Fmax=",Fclist[Fmax_arg])

        half_index = len(wlist) // 2
        for i in range(len(wlist)-2):
            w = wlist[i+1]
            grad_num = 0.5*(Fclist[i+2] - Fclist[i])/small_val
            u = w*costheta
            v = w*sintheta
            if False:
                fid1 = 2.0*(costheta*math.exp(-u*u) + sintheta*math.exp(-v*v))/((costheta + sintheta)*sqrt_pi*w)
                fid2 = (math.erf(u) + math.erf(v))/((costheta + sintheta)*w*w)
                grad_fla = fid1 - fid2
                if not test_run:
                    print("Gradient numerical",grad_num, "formula",grad_fla,"parts",fid1,fid2)

            if i == half_index-1:
                curv_num = (Fclist[i+2] + Fclist[i] - 2.0*Fclist[i+1])/(small_val*small_val)
                curv_fla = -4.0*(costheta ** 3 + sintheta ** 3)/(sqrt_pi*(costheta + sintheta))
                if not test_run:
                    print("Curvature numerical",curv_num, "formula",curv_fla,"ratio",curv_fla/curv_num)

        normal_high_list = normal_mfv(wlist, sigma=F_canon_sigma_high, offset=F_canon_offset_high, scale=F_canon_scale_high)
        normal_low_list = normal_mfv(wlist, sigma=F_canon_sigma_low, offset=F_canon_offset_low, scale=F_canon_scale_low)
        normal_low_arr = np.array(normal_low_list)

        curv = Fclist[half_index-1]+Fclist[half_index+1]-2.0*Fclist[half_index]
        curv_ref = normal_low_arr[half_index-1]+normal_low_arr[half_index+1]-2.0*normal_low_arr[half_index]
        if not test_run:
            print("Curvature ",curv,"ref",curv_ref,"ratio",curv_ref/curv)
        
        plt.close("all")
        plt.figure(num=1, dpi=240)
        ax = plt.gca()
        #plt.box(False)
        #ax.set_xlim((x_min, x_max))
        ax.set_ylim((0.0, 1.2))
        plt.axline((0.0, 0.0), (0.0, 0.5), color = "green", linewidth=1)
        plt.plot(wlist, Fclist, lw = 1.0, label="F_canon")
        plt.plot(wlist, normal_high_list, color = "red", lw = 1.0, label="normal high")
        plt.plot(wlist, normal_low_list, lw = 1.0, label="normal low")
        plt.legend()
        if not test_run:
            plt.show()

def check_line_data(test_run:bool, output_folder:str):
    Ntot = 1000
    D = 5.0
    alpha = 0.1
    Nb = int(alpha*Ntot) # bad
    Ng = int((1.0-alpha)*Ntot) # good
    ab = -0.5
    bb =  0.3
    sigma = 2.0

    # bad data
    xb = np.zeros(Nb)
    yb = np.zeros(Nb)
    for i in range(Nb):
        xb[i] = -D + 2.0*alpha*i*D/(Nb-1)
        yb[i] = ab*xb[i] + bb

    xb_last = xb[Nb-1]

    # good data
    xg = np.zeros(Ng)
    yg = np.zeros(Ng)
    for i in range(Ng):
        xg[i] = xb_last + (D-xb_last)*i/(Ng-1)
        yg[i] = 0.0

    def F_bad(a: float, b:float):
        tot = 0.0
        for i in range(Nb):
            r = a*xb[i] + b - yb[i]
            tot += math.exp(-0.5*r*r/(sigma*sigma))

        return tot

    def F_good(a: float, b:float):
        tot = 0.0
        for i in range(Ng):
            r = a*xg[i] + b - yg[i]
            tot += math.exp(-0.5*r*r/(sigma*sigma))

        return tot

    def F_bad_2(a: float, b:float):
        return Nb*sigma*sqrt_pi*(math.erf(((ab-a)*xb_last+bb-b)/(sqrt_2*sigma)) - math.erf((-(ab-a)*D+bb-b)/(sqrt_2*sigma)))/((D+xb_last)*(ab-a)*sqrt_2)

    def F_good_2(a: float, b:float):
        return Ng*sigma*sqrt_pi*(math.erf((a*D+b)/(sqrt_2*sigma)) - math.erf((a*xb_last+b)/(sqrt_2*sigma)))/((D-xb_last)*a*sqrt_2)

    def F_bad_canon(a: float, b:float):
        u = -((a-ab)*D+bb-b)/(sqrt_2*sigma)
        v = ((ab-a)*xb_last+bb-b)/(sqrt_2*sigma)
        return Nb*F_canon(u,v)*0.5*sqrt_pi

    def F_good_canon(a: float, b:float):
        u = -(a*D+b)/(sqrt_2*sigma)
        v = (a*xb_last+b)/(sqrt_2*sigma)
        return Ng*F_canon(u,v)*0.5*sqrt_pi

    # bad data check
    for b in np.linspace(-3.0, 3.0, 30):
        for a in np.linspace(-3.0, 3.0, 30):
            F_bab = F_bad(a,b)
            F_b2ab = F_bad_2(a,b)
            F_bcab = F_bad_canon(a,b)
            if not test_run:
                print("Bad ab=",a,b,"check ", F_bab, F_b2ab, "diff=",F_bab-F_b2ab,"F_bad_canon=",F_bcab,"ratio=",F_bab/F_bcab)

    # good data check
    for b in np.linspace(-3.0, 3.0, 30):
        for a in np.linspace(-3.0, 3.0, 30):
            F_gab = F_good(a,b)
            F_g2ab = F_good_2(a,b)
            F_gcab = F_good_canon(a,b)
            if not test_run:
                print("Good ab=",a,b,"check ", F_gab, F_g2ab, "diff=",F_gab-F_g2ab,"F_good_canon=",F_gcab,"ratio=",F_gab/F_gcab)

def build_cross_sections(test_run:bool, output_folder:str):
    Ntot = 1000
    D = 5.0
    alpha = 0.4
    Nb = int(alpha*Ntot) # bad
    Ng = int((1.0-alpha)*Ntot) # good
    xb_last = -D + 2.0*alpha*D
    sigma = 0.4
    for bb in np.linspace(-1.0, 1.0, 4 if test_run else 10):
        for ab in np.linspace(-0.3, 0.3, 6 if test_run else 20):
            def F_bad_canon(a: float, b:float):
                u = -((a-ab)*D+bb-b)/(sqrt_2*sigma)
                v = ((ab-a)*xb_last+bb-b)/(sqrt_2*sigma)
                return Nb*F_canon(u,v)*0.5*sqrt_pi

            def F_good_canon(a: float, b:float):
                u = -(a*D+b)/(sqrt_2*sigma)
                v = (a*xb_last+b)/(sqrt_2*sigma)
                return Ng*F_canon(u,v)*0.5*sqrt_pi

            xlist = np.linspace(-5.0, 5.0, 201)
            ylist = np.zeros(len(xlist))
            glist = np.zeros(len(xlist))
            blist = np.zeros(len(xlist))
            norm = math.sqrt(ab*ab + bb*bb)
            acoef = ab/norm
            bcoef = bb/norm
            for i,x in enumerate(xlist):
                a = x*acoef
                b = x*bcoef
                ylist[i] = F_good_canon(a, b) + F_bad_canon(a, b)
                glist[i] = F_good_canon(a, b)
                blist[i] = F_bad_canon(a, b)

            plt.close("all")
            plt.figure(num=1, dpi=240)
            #ax = plt.gca()
            #plt.box(False)
            #ax.set_xlim((x_min, x_max))
            #ax.set_ylim((0.0, 1.2))
            # a = x*a/norm, b = x*b/norm, so x = norm
            abx = norm
            plt.axline((0.0, 0.0), (0.0, 0.5), color = "yellow", linewidth=1)
            plt.axline((abx, 0.0), (abx, 0.5), color = "magenta", linewidth=1)
            plt.plot(xlist, ylist, lw = 1.0, label="F" + str(ab) + " b=" + str(bb), color = "blue")
            plt.plot(xlist, glist, lw = 1.0, label="Good", color = "green")
            plt.plot(xlist, blist, lw = 1.0, label="Bad", color = "red")
            plt.legend()
            if not test_run:
                plt.show()
                
def main(test_run:bool, output_folder:str="../../output"):
    output_folder += "/line_fit"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    if test_run:
        check_F_canon(0.0, test_run, output_folder)
        #check_F_canon(0.1, test_run, output_folder)
        #check_F_canon(0.5, test_run, output_folder)
        check_F_canon(5.0, test_run, output_folder)
        visualise_F_canon(test_run, output_folder)
        show_F_canon(test_run, output_folder)
        check_line_data(test_run, output_folder)

    build_cross_sections(test_run, output_folder)

    if test_run:
        print("compare_erf_uv_normal OK")

if __name__ == "__main__":
    main(False) # test_run
