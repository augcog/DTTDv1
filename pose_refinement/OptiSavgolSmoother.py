import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))


class OptiSavgolSmoother():
    @staticmethod
    def smooth_opti_poses_savgol(scene_dir, pose_df, write_smoothed_to_file=False):
        
        pose_df = pose_df.replace('', np.NaN).astype(np.float32)
        pose_df = pose_df.interpolate()

        xyzs = np.array(pose_df[['camera_Position_X', 'camera_Position_Y', 'camera_Position_Z']]).astype(np.float32)
        rots = np.array(pose_df[['camera_Rotation_X', 'camera_Rotation_Y', 'camera_Rotation_Z', 'camera_Rotation_W']]).astype(np.float32)

        x = xyzs[:,0]
        y = xyzs[:,1]
        z = xyzs[:,2]
        a = rots[:,0]
        b = rots[:,1]
        c = rots[:,2]
        d = rots[:,3]

        window_length = 21
        polyorder = 3

        res_x = signal.savgol_filter(x, window_length, polyorder)
        res_y = signal.savgol_filter(y, window_length, polyorder)
        res_z = signal.savgol_filter(z, window_length, polyorder)
        res_a = signal.savgol_filter(a, window_length, polyorder)
        res_b = signal.savgol_filter(b, window_length, polyorder)
        res_c = signal.savgol_filter(c, window_length, polyorder)
        res_d = signal.savgol_filter(d, window_length, polyorder)

        smoothed_pose_df = pose_df.copy()
        smoothed_pose_df['camera_Position_X'] = res_x
        smoothed_pose_df['camera_Position_Y'] = res_y
        smoothed_pose_df['camera_Position_Z'] = res_z
        smoothed_pose_df['camera_Rotation_X'] = res_a
        smoothed_pose_df['camera_Rotation_Y'] = res_b
        smoothed_pose_df['camera_Rotation_Z'] = res_c
        smoothed_pose_df['camera_Rotation_W'] = res_d

        if write_smoothed_to_file:
            output_file = os.path.join(scene_dir, 'camera_poses', 'camera_poses_smoothed.csv')
            smoothed_pose_df.to_csv(output_file)

        return smoothed_pose_df