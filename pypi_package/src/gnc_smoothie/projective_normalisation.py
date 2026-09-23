import math
import numpy as np
import numpy.typing as npt

class ProjectiveNormalisation:
    def __init__(
        self,
        *,
        max_iterations:int = 100,
        term_threshold:float = 1.e-12
    ):
        self.__max_iterations = max_iterations
        self.__term_threshold = term_threshold
        self.error_string = None

    def run(self, data: npt.ArrayLike):
        assert(data.ndim == 2 or data.ndim == 3)
        if data.ndim == 2:
            data = data.reshape((data.shape[0],1,data.shape[1]))
            data_reshaped = True
        else:
            data_reshaped = False

        S = np.identity(data.shape[2])
        for itn in range(self.__max_iterations):
            # compute S^-1
            Si = np.linalg.inv(S)

            # compute Sp = sum of (B_i^T*B_i) / ||B_i*S^-1*B_i^T||_F where _F
            # denotes the Frobenius norm of a matrix
            Sp = np.zeros((data.shape[2],data.shape[2]))
            for B in data:
                # compute BTB = B_i^T*B_i
                BT = np.transpose(B)
                BTB = np.matmul(BT, B)

                # compute BSiBT = B_i*S^-1*B_i^T
                BSiBT = np.matmul(B, np.matmul(Si, BT))

                # scale BTB using inverse of trace of BSiBT
                norm = np.linalg.trace(BSiBT)
                if norm <= 0:
                    self.error_string = "Trace of BTB is invalid"
                    break

                BTB /= norm

                # increment sum by adjusting Sp
                Sp += BTB

            if self.error_string is not None:
                break

            # scale new S (Sp) by its trace to normalise it
            norm = np.linalg.trace(Sp)
            if norm <= 0.0:
                self.error_string = "Trace of Sp is invalid"
                break

            Sp /= norm

            # break out of loop if the Frobenius norm of the difference between
            # the old S and new S (Sp) is smaller than a threshold
            if(np.linalg.norm(S-Sp) < self.__term_threshold):
                break

            # replace S by new solution Sp
            S = Sp
                
        # if all the iterations were completed, the algorithm didn't converge,
        # so return with failure
        if itn == self.__max_iterations:
            self.error_string = "No convergence"
        elif self.error_string is None:
            self.S = S

            # use Cholesky factorisation to compute U where U^T*U = S
            self.U = np.linalg.cholesky(self.S)

            # create normalised version of data
            self.ndata = np.copy(data)

            # right-multiply each matrix by U^-1 and scale to unit Frobenius norm
            UinvT = np.transpose(np.linalg.inv(self.U))
            for i,B in enumerate(self.ndata):
                B = np.linalg.matmul(B, UinvT)
                self.ndata[i] = B/np.linalg.norm(B)

            if data_reshaped:
                self.ndata = self.ndata.reshape((self.ndata.shape[0], self.ndata.shape[2]))
