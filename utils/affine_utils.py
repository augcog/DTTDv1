import numpy as np
from scipy.spatial.transform import Rotation as R

def invert_affine(affine):
    return np.linalg.inv(affine)

#quats of shape (4xN)
#return average quaternion

def average_quaternion(quats):
    assert(quats.shape[1] == 4)

    avg_quat = np.linalg.eigh(np.einsum('ij,ik,i->...jk', quats, quats, np.ones((quats.shape[0]))))[1][:, -1]

    return avg_quat

def affine_matrix_from_rotvec_trans(rot_vec, trans):
    rot = R.from_rotvec(rot_vec)
    rmatrix = rot.as_matrix()
    trans = np.expand_dims(trans, -1)
    aff = np.hstack((rmatrix, trans))
    lrow = np.array([0,0,0,1])
    aff = np.vstack((aff, lrow))
    return aff