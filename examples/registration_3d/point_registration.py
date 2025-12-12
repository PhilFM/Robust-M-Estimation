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
    #   = Rs*R0*x + t, Rs = ( 1  -az  ay), R0*x = (R0_xx*x_x + R0_xy*x_y + R0_xz*x_z) = (R0x_x)
    #                       ( az  1  -ax)         (R0_yx*x_x + R0_yy*x_y + R0_yz*x_z)   (R0x_y)
    #                       (-ay  ax  1 )         (R0_zx*x_x + R0_zy*x_y + R0_zz*x_z)   (R0x_z)
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
        R,t = LS_PointCloudRegistration(data[0], weight[0])
        model = np.zeros(6)
        model[3:6] = t
        return model,R

