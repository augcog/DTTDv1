FRONT_LOSS_COEFF = 1

import numpy as np

#quats of shape (Nx4)
#return average quaternion

def average_quaternion(quats):
    assert(quats.shape[1] == 4)

    avg_quat = np.linalg.eigh(np.einsum('ij,ik,i->...jk', quats, quats, np.ones((quats.shape[0]))))[1][:, -1]

    return avg_quat