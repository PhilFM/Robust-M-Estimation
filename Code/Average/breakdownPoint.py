import math

# sp = p*s, s^2 + sp^2 = sc^2
# So sc^2 = (1+p^2)*s^2
# So sc = sqrt(1+p^2)*s
# s/sc = 1/sqrt(1+p^2)
# breakdownpoint = 0.5*s^3*sc^-3*exp(0.5*sp^2/sc^2)
#                = 0.5*(1+p^2)^-3/2*exp(0.5*p^2/(1+p^2))
def breakdownPoint(p):
    return 0.5*math.pow(1.0+p*p, -1.5)*math.exp(0.5*p*p/(1.0+p*p))

for p in [0.2,0.4,0.5,0.7,0.8,1.0]:
    print(p, breakdownPoint(p))
