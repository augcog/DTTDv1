import os
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(np.identity(4), True)
    >>> np.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(np.diag([1, -1, -1, 1]))
    >>> np.allclose(q, [0, 1, 0, 0]) or np.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> np.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> np.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> np.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True
    >>> R = euler_matrix(0.0, 0.0, np.pi/2.0)
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q

pose_path = r'C:\Users\OpenARK\Desktop\adam\Left_right_left_starting_z_cleaned.csv'
cleaned_pose_path = pose_path[:pose_path.rfind(".csv")] + "_cleaned.csv"
corrected_pose_path = pose_path[:pose_path.rfind(".csv")] + "_corrected.csv"

poses = pd.read_csv(pose_path)
poses = poses.dropna()

rotations = poses[["camera_Rotation_X", "camera_Rotation_Y", "camera_Rotation_Z", "camera_Rotation_W"]]
rotations = np.array(rotations)

rotations = R.from_quat(rotations).as_matrix()

R_opti_to_ue4_real = np.array([[[-1, 0, 0], [0, -1, 0], [0, 0, 1]]])

P_ue4_real_to_ue4 = np.array([[[0, 0, 1], [1, 0, 0], [0, -1, 0]]])

rotations_corrected = np.matmul(rotations.transpose((0, 2, 1)), R_opti_to_ue4_real.transpose((0, 2, 1))).transpose((0, 2, 1))

rotations_corrected_quat = R.from_matrix(rotations_corrected).as_quat()

rotations_corrected_quat_pd = np.zeros_like(rotations_corrected_quat)
rotations_corrected_quat_pd[:,0] = rotations_corrected_quat[:,-2]
rotations_corrected_quat_pd[:,1] = rotations_corrected_quat[:,1]
rotations_corrected_quat_pd[:,2] = rotations_corrected_quat[:,0]
rotations_corrected_quat_pd[:,3] = rotations_corrected_quat[:,3]

rotations_corrected_quat_pd = rotations_corrected_quat

rotations_corrected = np.matmul(rotations_corrected.transpose((0, 2, 1)), P_ue4_real_to_ue4.transpose((0, 2, 1))).transpose((0, 2, 1))

rotations_corrected = rotations_corrected.reshape((-1, 9))

poses[["camera_Rotation_00", "camera_Rotation_01", "camera_Rotation_02", "camera_Rotation_10", "camera_Rotation_11", "camera_Rotation_12", "camera_Rotation_20", "camera_Rotation_21", "camera_Rotation_22"]] = rotations_corrected

poses[["camera_Rotation_X", "camera_Rotation_Y", "camera_Rotation_Z", "camera_Rotation_W"]] = rotations_corrected_quat_pd

poses.to_csv(corrected_pose_path)
