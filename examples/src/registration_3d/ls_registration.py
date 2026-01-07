import numpy as np

def LS_RotationSearch(Y, X, weight):
    # Y = R*X
    # Y: 3xm 
    # X: 3xm
    nPoints = len(Y)
    for i in range(0,nPoints):
        Y[i][0] *= weight[i]
        Y[i][1] *= weight[i]
        Y[i][2] *= weight[i]

    M = np.matmul(np.transpose(Y),X)
    #print("M=",M)

    U, S, Vt = np.linalg.svd(M, full_matrices=True)
    #print("U=",U,"S=",S,"Vt=",Vt)
    #Sd = np.zeros((3,3))
    #Sd[0][0] = S[0]
    #Sd[1][1] = S[1]
    #Sd[2][2] = S[2]
    #print("U=",U,"Sd=",Sd,"Vt=",Vt)
    #print("Reconstructed M=", np.matmul(np.matmul(U,Sd),Vt))

    D = np.zeros((3,3))
    D[0][0] = 1.0
    D[1][1] = 1.0
    D[2][2] = np.linalg.det(U)*np.linalg.det(Vt)

    #print("U=",U,"D=",D,"Vt=",Vt)
    return np.matmul(np.matmul(U,D),Vt)

def LS_PointCloudRegistration(data, weight):
    X = data[:,0]
    Y = data[:,1]
    #print("data=",data)
    #print("X=",X)
    #print("Y=",Y)

    s = sum(weight)

    x = np.matmul(weight,X)/s
    y = np.matmul(weight,Y)/s
    #print("s=",s,"x=",x,"y=",y)

    X_ = X - x
    Y_ = Y - y
    #print("X_=",X_,"Y_=",Y_)

    R= LS_RotationSearch(Y_, X_, weight)

    t = y - np.matmul(R,x)
    return R,t

