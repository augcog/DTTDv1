"""
Use Kalman Filter to smooth optitrack poses.
"""

import numpy as np
import pandas as pd
from pykalman import UnscentedKalmanFilter

import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from utils.constants import OPTI_FRAMERATE
from utils.pose_dataframe_utils import pose_df_from_xyzs_rots

class OptiKFSmoother():
    def __init__(self):
        pass

    dt = 1. / OPTI_FRAMERATE

    @staticmethod
    def transition_fn(state, noise):
        next_state = np.zeros_like(state)
        x, y, z, dx, dy, dz, a, b, c, d, wx, wy, wz = state
        next_state[0] = x + dx * OptiKFSmoother.dt
        next_state[1] = y + dy * OptiKFSmoother.dt
        next_state[2] = z + dz * OptiKFSmoother.dt
        next_state[3] = dx
        next_state[4] = dy
        next_state[5] = dz
        next_state[6] = a + OptiKFSmoother.dt / 2. * wx * a
        next_state[7] = b + OptiKFSmoother.dt / 2. * wy * b
        next_state[8] = c + OptiKFSmoother.dt / 2. * wz * c
        next_state[9] = d
        next_state[10] = wx
        next_state[11] = wy
        next_state[12] = wz

        return next_state + noise
    
    @staticmethod
    def observation_fn(state, noise):
        obs = np.array([state[0], state[1], state[2], state[6], state[7], state[8], state[9]])
        return obs + noise

    @staticmethod
    def smooth_opti_poses_kf(scene_dir, pose_df, write_smoothed_to_file=False):
        random_state = np.random.RandomState(0)
        initial_state_covariance = np.eye(13) * 0.01
        transition_covariance = np.eye(13) * 0.1
        observation_covariance = np.eye(7) + random_state.randn(7, 7) * 0.1

        xyzs = np.array(pose_df[['camera_Position_X', 'camera_Position_Y', 'camera_Position_Z']]).astype(np.float32)
        rots = np.array(pose_df[['camera_Rotation_X', 'camera_Rotation_Y', 'camera_Rotation_Z', 'camera_Rotation_W']]).astype(np.float32)

        poses = np.hstack((xyzs, rots))

        initial_pose = poses[0]
        initial_state = np.zeros((13))
        initial_state[0] = initial_pose[0]
        initial_state[1] = initial_pose[1]
        initial_state[2] = initial_pose[2]
        initial_state[6] = initial_pose[3]
        initial_state[7] = initial_pose[4]
        initial_state[8] = initial_pose[5]
        initial_state[9] = initial_pose[6]

        kf = UnscentedKalmanFilter(OptiKFSmoother.transition_fn, OptiKFSmoother.observation_fn, 
        transition_covariance, observation_covariance, initial_state, initial_state_covariance,
        random_state=random_state)

        smoothed_poses = kf.smooth(poses)[0]

        xyzs = smoothed_poses[:,0:3]
        quats = smoothed_poses[:,6:10]

        smoothed_pose_df = pose_df.copy()
        smoothed_pose_df[['camera_Position_X', 'camera_Position_Y', 'camera_Position_Z']] = xyzs
        smoothed_pose_df[['camera_Rotation_X', 'camera_Rotation_Y', 'camera_Rotation_Z', 'camera_Rotation_W']] = quats

        if write_smoothed_to_file:
            output_file = os.path.join(scene_dir, 'camera_poses', 'camera_poses_smoothed.csv')
            smoothed_pose_df.to_csv(output_file)
        
        return smoothed_pose_df