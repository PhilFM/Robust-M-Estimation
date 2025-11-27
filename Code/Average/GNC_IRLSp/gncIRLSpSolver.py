import numpy as np
import matplotlib.pyplot as plt
import sys

from GNC_IRLSpMean import GNC_IRLSpMean

sys.path.append("../../Library")
from IRLS import IRLS
from GNC_IRLSpParams import GNC_IRLSpParams

# configuration
showSolution = True
showGradient = False

# sigma  0.2 - good
#        0.8 - too large
#       0.02 - too small
p = 0.0
rscale = 1.0
epsilonBase = 0.2
epsilonLimit = 1.0
beta = 0.8

# override x limit 
#xMin = 0.7
#xMax = 1.3
xMin = xMax = None

# data is a list of [weight, value] pairs

if showGradient:
    data = np.array([0.0, # good data
                     0.25, 0.1, -0.2, -0.3]) # bad data
    weight = np.array([5.0, # good data
                       1.0, 1.0, 1.0, 1.0]) # bad data
else:
    data = np.array([0.88, 0.93, 1.0, 1.06, 1.1, # good data
                     10.0, 2.0, 2.5, 3.2]) # bad data
    weight = np.array([1.0, 1.0, 1.0, 1.0, 1.0, # good data
                       1.0, 1.0, 1.0, 1.0]) # bad data

paramInstance = GNC_IRLSpParams(p, rscale, epsilonBase, epsilonLimit, beta)
algInstance = GNC_IRLSpMean(paramInstance, data, weight)
m = IRLS(algInstance, printWarnings=True).run()
print("Result: m=", m)

# get min and max of data
yMin = yMax = 0.0

if xMin == None:
    if showGradient:
        xMin = -2.0*sigma
        xMax =  2.0*sigma
    else:
        dmin = dmax = data[0]
        for d in data:
            dmin = min(dmin, d)
            dmax = max(dmax, d)
            print("d=", d, " min/max=", dmin, dmax)

        # allow border
        drange = dmax-dmin
        xMin = dmin - 0.05*drange
        xMax = dmax + 0.05*drange

print("xMin=", xMin, " xMax=", xMax)
mlist = np.linspace(xMin, xMax, num=300)

def objectiveFunc(m):
    return algInstance.objectiveFunc([m])

def gradientFunc(m):
    return algInstance.gradient([m])[0]

if showGradient:
    for mx in mlist:
        #print("x=", mx, " grad=", gradientFunc(mx, sigma, data))
        yMin = min(yMin, gradientFunc(mx))
        yMax = max(yMax, gradientFunc(mx))
else:
    for mx in mlist:
        yMax = max(yMax, objectiveFunc(mx))

print("yMin=", yMin, " yMax=", yMax)
yMin *= 1.01 # allow for a small border
yMax *= 1.01 # allow for a small border

plt.figure(num=1, dpi=240)
ax = plt.gca()
#plt.box(False)
ax.set_ylim((yMin, yMax))

if showGradient:
    rmgv = np.vectorize(gradientFunc)
    plt.plot(mlist, rmgv(mlist), lw = 1.0)
    for d,w in zip(data,weight, strict=True):
        if d >= xMin and d <= xMax:
            plt.axvline(x = d, color = 'b', ymax = 0.05*w, lw = 1.0)

    #fig.gca().set_ylabel(r'$\lambda$')
    plt.axhline(y = 0.0, color = 'b', label = '', lw = 1.0)
    plt.axvline(x = -sigma, color = 'g', label = r'x=-$\sigma$', lw = 1.0)
    plt.axvline(x =  sigma, color = 'r', label = r'x= $\sigma$', lw = 1.0)
    if showSolution:
        plt.axvline(x = m[0], color = 'r', label = 'solution', lw = 1.0)
else:
    hmfv = np.vectorize(objectiveFunc)
    plt.plot(mlist, hmfv(mlist), lw = 1.0)
    for d,w in zip(data,weight, strict=True):
        if d >= xMin and d <= xMax:
            plt.axvline(x = d, color = 'b', ymax = 0.1*w, lw = 1.0)

    if showSolution:
        plt.axvline(x = m[0], color = 'r', label = 'solution', lw = 1.0)

plt.legend()
plt.savefig('../../../Output/gncIRLSpMean.png', bbox_inches='tight')
plt.show()
