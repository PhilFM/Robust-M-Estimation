import math
import sys
import matplotlib.pyplot as plt

sys.path.append("Welsch")
from WelschMean import WelschMean

sys.path.append("../Library")
from GNC_WelschParams import GNC_WelschParams

def g(p):
    p2p1 = 1.0 + p*p
    D = math.sqrt(1.0 + 2.0*p*p + 1.25*p*p*p*p)
    fid = p*p + 2.0*D
    fid2 = (2.0*p2p1/fid)
    fid3 = math.exp(0.125*fid*fid/(p2p1*p2p1))
    return fid2*fid3

def maxSolution(p, sigma):
    sigmap = sigma*p
    sigmac = math.sqrt(sigma*sigma + sigmap*sigmap)

    diffs = sigma*sigma-sigmac*sigmac
    sigma4 = sigma*sigma*sigma*sigma
    sigmac4 = sigmac*sigmac*sigmac*sigmac
    disc = math.sqrt(diffs*diffs + 4.0*sigmac4)
    y = sigma*(diffs - disc)/(2.0*sigmac*sigmac)
    k = -sigma4*math.exp(-0.5*sigma*sigma/(sigmac*sigmac))*math.exp(0.5*y*y/(sigma*sigma))/(sigmac*sigmac*sigmac*y)
    
    p2p1 = 1.0 + p*p
    yp = 0.5*(-p*p - math.sqrt(p*p*p*p + 4.0*p2p1*p2p1))/p2p1
    kp = -math.exp(-0.5/p2p1)*math.exp(0.5*yp*yp)/(yp*math.pow(p2p1,1.5))
    #print("yp=",yp, " ypapprox=",-1.0-0.5*p*p, "yp2=",y/sigma)
    z = sigma-y
    #print("y=",y, " z=",z)

    D = math.sqrt(1.0 + 2.0*p*p + 1.25*p*p*p*p)
    ypp = 0.5*(-p*p - 2.0*D)/p2p1
    kpp = -math.pow(p2p1,-1.5)*math.exp(-0.5/p2p1)*math.exp(0.5*ypp*ypp)/ypp
    #print("k=",k," kpp=",kpp)
    f1 = -math.exp(-0.5/p2p1)*math.exp(0.5*ypp*ypp)/ypp
    fid = p*p + 2.0*D
    f1p = (2.0*p2p1/fid)*math.exp(-0.5/p2p1)*math.exp(0.125*fid*fid/(p2p1*p2p1))
    f2 = math.exp(0.5*p*p/p2p1)
    #print("f1=",f1,"f1p=",f1p," f2=",f2, " diff=", f1-f2)

    fid2 = (2.0*p2p1/fid)
    fid3 = math.exp(0.125*fid*fid/(p2p1*p2p1))
    g1 = fid2*fid3
    g2 = math.exp(0.5)
    g1p = g(p)
    print("fid2=",fid2," fid3=",fid3," g1=",g1," g2=",g2, " diff=", g1-g2, " check one:", g1*f2/(g2*f1), " g1p=",g1p)

    g1deriv = p*p*p*math.exp(0.125*fid*fid/(p2p1*p2p1))/(p2p1*p2p1*D)
    smallVal = 1.0e-5
    g1derivNum = 0.5*(g(p+smallVal)-g(p-smallVal))/smallVal
    print("g1deriv=",g1deriv," g1derivNum=",g1derivNum)

    return z, k, kp, sigmac

sigma = 2.0
for p in [0.0,0.01,0.03,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5,2.0,2.5,10.0,100.0]:
    z,k,kp,sigmac = maxSolution(p, sigma)
    print("p=",p," z=",z," k=",k," kp=",kp," sigmac=",sigmac)
    sdiv = sigma/sigmac
    print("s=",sdiv," sdiv2=",sdiv*sdiv," sdiv3=",sdiv*sdiv*sdiv, "sdiv3p=",sdiv*sdiv*sdiv*math.exp(0.5*p*p/(1.0+p*p)))

    goodData = [0.0]
    goodWeight = [sigma/sigmac]
    outlierData = [z]
    outlierWeight = [k]

    paramInstance = GNC_WelschParams(sigma)
    goodWelschMeanInstance = WelschMean(paramInstance, goodData, goodWeight)
    outlierWelschMeanInstance = WelschMean(paramInstance, outlierData, outlierWeight)

    goodGradient = goodWelschMeanInstance.gradient([sigma])[0]
    outlierGradient = outlierWelschMeanInstance.gradient([sigma])[0]
    print("Gradients: ", goodGradient, outlierGradient)

    good2ndDeriv = goodWelschMeanInstance.secondDeriv([sigma])[0][0]
    outlier2ndDeriv = outlierWelschMeanInstance.secondDeriv([sigma])[0][0]
    print("2nd derivatives: ", good2ndDeriv, outlier2ndDeriv)

plt.figure(num=1, dpi=240)
ax = plt.gca()

plt.legend()
plt.savefig('breakdownPoint2.png', bbox_inches='tight')
plt.show()
