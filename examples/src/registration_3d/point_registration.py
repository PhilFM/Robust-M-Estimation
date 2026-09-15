import numpy as np
from scipy.spatial.transform import Rotation as Rot

from ls_registration import LS_PointCloudRegistration

class PointRegistration:
    def __init__(self):
        pass

    # copy model parameters and apply any internal calculations
    def cache_model(self, model, model_ref=None):
        rotd = Rot.from_mrp(-0.25*model[0:3])
        self.__R = np.matmul(Rot.as_matrix(rotd), model_ref)
        self.__t = model[3:6]

    # r = y - R*x - t
    #   = Rs*R0*x + t, Rs = ( 1-(ay^2+az^2)/2   -az+ax*ay/2      ay+az*ax/2   ),    R0*x = (R0_xx*x_x + R0_xy*x_y + R0_xz*x_z) = (R0x_x)
    #                       (    az+ax*ay/2   1-(az^2+ax^2)/2   -ax+ay*az/2   )            (R0_yx*x_x + R0_yy*x_y + R0_yz*x_z)   (R0x_y)
    #                       (   -ay+az*ax/2      ax+ay*az/2   1-(ax^2+ay^2)/2 )            (R0_zx*x_x + R0_zy*x_y + R0_zz*x_z)   (R0x_z)
    # where R0x = R0*x
    def residual(self, data_item) -> np.array:
        x = data_item[0]
        y = data_item[1]
        return np.array(y - np.matmul(self.__R,x) - self.__t)

    # dr   (  0     R0x_z -R0x_y)          dr
    # -- = (-R0x_z   0     R0x_x) = Rx_x,  -- = -I_3x3
    # da   ( R0x_y -R0x_x   0   )          dt
    def residual_gradient(self, data_item) -> np.array:
        x = data_item[0]
        Rx = np.matmul(self.__R,x)
        return np.array([[   0.0,  Rx[2], -Rx[1], -1.0,  0.0,  0.0],
                         [-Rx[2],    0.0,  Rx[0],  0.0, -1.0,  0.0],
                         [ Rx[1], -Rx[0],    0.0,  0.0,  0.0, -1.0]])

    # d2rx   (0        R0x_y/2  R0x_z/2)  d2ry   (-R0x_y   R0x_x/2     0   )  d2rz   (-R0x_z       0    R0x_x/2)
    # ---- = (R0x_y/2 -R0x_x       0   ), ---- = ( R0x_x/2    0     R0x_z/2), ---- = (    0    -R0x_z   R0x_y/2)
    #  da2   (R0x_z/2    0     -R0x_x  )   da2   (   0     R0x_z/2 -R0x_y  )   da2   ( R0x_x/2  R0x_y/2    0   )
    def residual_2nd_deriv(self, data_item) -> np.array:
        x = data_item[0]
        Rx = np.matmul(self.__R,x)
        arr = np.zeros((3,6,6))
        arr[1][0][1] = arr[1][1][0] = arr[2][0][2] = arr[2][2][0] = 0.5*Rx[0]
        arr[0][0][1] = arr[0][1][0] = arr[2][1][2] = arr[2][2][1] = 0.5*Rx[1]
        arr[0][0][2] = arr[0][2][0] = arr[1][1][2] = arr[1][2][1] = 0.5*Rx[2]
        arr[0][1][1] = arr[0][2][2] = -Rx[0]
        arr[1][2][2] = arr[1][0][0] = -Rx[1]
        arr[2][0][0] = arr[2][1][1] = -Rx[2]
        return -arr

    def update_model_ref(self, model, prev_model_ref=None):
        rotd = Rot.from_mrp(-0.25*model[0:3])
        if prev_model_ref is None:
            R = Rot.as_matrix(rotd)
        else:
            R = np.matmul(Rot.as_matrix(rotd), prev_model_ref)

        # reset model parameters because they are subsumed by reference
        model[0:3] = 0.0

        # convert to quaternion and back to matrix to ensure orthogonality
        q = Rot.as_quat(Rot.from_matrix(R))
        return Rot.as_matrix(Rot.from_quat(q))

    # fits the model to the data
    def weighted_fit(self, data, weight, scale) -> (np.array, np.array):
        R,t = LS_PointCloudRegistration(data, weight)
        model = np.zeros(6)
        model[3:6] = t
        return model,R

