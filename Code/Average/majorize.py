import math
import numpy as np
import matplotlib.pyplot as plt

def refFunc(r):
    return 1.0 - math.exp(-0.5*r*r)

def majFunc(r):
    return 0.5*r*r*math.exp(-0.5*r*r)

rlist = np.linspace(-3.0, 3.0, num=100)
refFuncV = np.vectorize(refFunc)
majFuncV = np.vectorize(majFunc)
plt.figure(num=1, dpi=240)

ax = plt.gca()

ax.set_ylim((0.0, 1.0))
#ax.set_ylabel("f(y)")
ax.set_xlabel("y")

plt.plot(rlist, rlist*0, "--", label="")
plt.plot(rlist, refFuncV(rlist), lw = 1.0, color = 'blue', label="Ref")
plt.plot(rlist, majFuncV(rlist), lw = 1.0, color = 'green', label="Maj")

plt.legend()
plt.savefig('majorize.png', bbox_inches='tight')
plt.show()

def welschMajorizeFunc(x):
    return -1.0 + math.exp(x) - x

majFuncV = np.vectorize(welschMajorizeFunc)

ax = plt.gca()
ax.set_ylim((0.0, 16.0))
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$f(x)$")

plt.plot(rlist, majFuncV(rlist), lw = 1.0, color = 'green') #, label="Welsch majorize test")

plt.legend()
plt.savefig('../Doc/RobustAverage/pictures/welschMajorizeTest.png', bbox_inches='tight')
plt.show()
