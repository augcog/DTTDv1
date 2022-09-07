import numpy as np
from scipy.spatial.transform import Rotation as R

def invert_affine(affine):
    return np.linalg.inv(affine)

#quats of shape (Nx4)
#return average quaternion

def average_quaternion(quats):
    assert(quats.shape[1] == 4)

    avg_quat = np.linalg.eigh(np.einsum('ij,ik,i->...jk', quats, quats, np.ones((quats.shape[0]))))[1][:, -1]

    return avg_quat

def affine_matrix_from_rotmat_trans(rotmat, trans):
    aff = np.hstack((rotmat, trans))
    lrow = np.array([0, 0, 0, 1])
    aff = np.vstack((aff, lrow))
    return aff

def affine_matrix_from_rotvec_trans(rot_vec, trans):
    rot = R.from_rotvec(rot_vec)
    rmatrix = rot.as_matrix()
    trans = np.expand_dims(trans, -1)
    aff = np.hstack((rmatrix, trans))
    lrow = np.array([0,0,0,1])
    aff = np.vstack((aff, lrow))
    return aff

def rotvec_trans_from_affine_matrix(aff):
    rot_mat = aff[:3,:3]
    trans = aff[:3,3]
    rotvec = R.from_matrix(rot_mat).as_rotvec()
    return rotvec, trans

def affine_matrix_from_rot_mat(rot_mat):
    aff = np.eye(4)
    aff[:3,:3] = rot_mat
    return aff

def affine_matrix_from_trans(trans):
    aff = np.eye(4)
    aff[:3,3] = trans
    return aff