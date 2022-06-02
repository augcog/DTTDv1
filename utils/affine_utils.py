import numpy as np
from scipy.spatial.transform import Rotation as R

def invert_affine(affine):
    assert(affine.shape == (4, 4))

    invert_affine = np.zeros((4, 4))
    invert_affine[:3,:3] = affine[:3,:3].T
    invert_affine[:3,3] = -affine[:3,3]
    invert_affine[3,3] = 1.

    return invert_affine

#quats of shape (4xN)
#return average quaternion

def average_quaternion(quats):
    assert(quats.shape[0] == 4)

    q = quats @ quats.T

    w, v = np.linalg.eig(q)

    max_eigenvalue_idx = np.argmax(w)
    return v[:,max_eigenvalue_idx]


def affine_matrix_from_rotvec_trans(rot_vec, trans):
    rot = R.from_rotvec(rot_vec)
    rmatrix = rot.as_matrix()
    trans = np.expand_dims(trans, -1)
    aff = np.hstack((rmatrix, trans))
    lrow = np.array([0,0,0,1])
    aff = np.vstack((aff, lrow))
    return aff