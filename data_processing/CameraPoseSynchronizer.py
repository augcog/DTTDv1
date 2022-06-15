import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.interpolate import Akima1DInterpolator
from scipy.spatial.transform import Rotation as R
import yaml

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from calculate_extrinsic.CameraOptiExtrinsicCalculator import CameraOptiExtrinsicCalculator
from utils.camera_utils import load_intrinsics, load_distortion
from utils.frame_utils import load_bgr, write_bgr, load_depth, write_depth

class CameraPoseSynchronizer():
    def __init__(self):
        pass

    @staticmethod
    def load_from_file(synchronized_poses_path):
        df = pd.read_csv(synchronized_poses_path)
        return df

    """

    Uses ARUCO data from frames and opti poses to synchronize the two pose streams.
    Writes a cleaned data and cleaned camera_poses_synchronized.csv (numbered 00000-n)

    Returns DF:
    Frame, camera_Rotation_X,camera_Rotation_Y,camera_Rotation_Z,camera_Rotation_W,camera_Position_X,camera_Position_Y,camera_Position_Z
    """
    @staticmethod
    def synchronize_camera_poses_and_frames(scene_dir, cleaned_opti_poses, show_sync_plot=True, write_to_file=False):

        camera_data_csv = os.path.join(scene_dir, "camera_data.csv")
        camera_time_break_csv = os.path.join(scene_dir, "camera_time_break.csv")
        scene_metadata_file = os.path.join(scene_dir, "scene_meta.yaml")

        with open(scene_metadata_file, 'r') as file:
            scene_metadata = yaml.safe_load(file)

        camera_name = scene_metadata["camera"]
        
        rotation_x_key = '{0}_Rotation_X'.format(camera_name)
        rotation_y_key = '{0}_Rotation_Y'.format(camera_name)
        rotation_z_key = '{0}_Rotation_Z'.format(camera_name)
        rotation_w_key = '{0}_Rotation_W'.format(camera_name)
        position_x_key = '{0}_Position_X'.format(camera_name)
        position_y_key = '{0}_Position_Y'.format(camera_name)
        position_z_key = '{0}_Position_Z'.format(camera_name)

        camera_intrinsics = load_intrinsics(camera_name)
        camera_distortion = load_distortion(camera_name)

        raw_frames_dir = os.path.join(scene_dir, "data_raw")

        time_break = pd.read_csv(camera_time_break_csv)
        calibration_start_frame_id, capture_start_frame_id = time_break["Calibration Start ID"].iloc[0], time_break["Capture Start ID"].iloc[0]

        camera_df = pd.read_csv(camera_data_csv)
        camera_timestamp = camera_df['Timestamp']
        camera_df['converted_camera_timestamp'] = pd.to_datetime(camera_timestamp, format='%Y-%m-%d %H:%M:%S.%f')

        #calculate virtual -> opti from ARUCO and extrinsic

        camera_opti_calc = CameraOptiExtrinsicCalculator()
        aruco_to_opti = camera_opti_calc.get_aruco_to_opti()

        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        parameters =  cv2.aruco.DetectorParameters_create()

        def calculate_virtual_to_opti(row):
            
            color_image = load_bgr(raw_frames_dir, row["Frame"])

            corners, ids, rejectedCandidates = cv2.aruco.detectMarkers(color_image, dictionary, parameters=parameters)
            xyz_pos = np.array([0, 0, 0])
            if np.all(ids is not None):  # If there are markers found by detector
                assert(len(ids) == 1)

                # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[0], 0.02, camera_intrinsics,
                                                                            camera_distortion)
                rvec = np.squeeze(rvec, axis = 1)
                tvec = np.squeeze(tvec, axis = 1)
                markerPoints = np.squeeze(markerPoints, axis=1)
                rotmat = R.from_rotvec(rvec).as_matrix() # rvec -> rotation matrix
                aruco_to_sensor = np.zeros((4,4)) # aruco -> sensor
                aruco_to_sensor[:3, :3] = rotmat
                aruco_to_sensor[:3, 3] = tvec * 9
                aruco_to_sensor[3,3] = 1
                sensor_to_aruco = np.linalg.inv(aruco_to_sensor) # sensor -> aruco 
                sensor_to_opti = aruco_to_opti @ sensor_to_aruco # sensor -> opti
                xyz_pos = sensor_to_opti[:3, -1]
                return xyz_pos

            else:
                x = np.array([0, 0, 0]).astype(np.float64)
                return x

        camera_first_timestamp = camera_df['converted_camera_timestamp'].iloc[0]

        camera_calib_df = camera_df.copy()
        camera_calib_df = camera_calib_df[camera_calib_df['Frame'].apply(lambda x : x >= calibration_start_frame_id and x < capture_start_frame_id)]
        aruco_computed_virtual_to_opti = np.array(camera_calib_df.apply(calculate_virtual_to_opti, axis=1, result_type="expand")).astype(np.float64)
        
        camera_calib_df["position_x"] = aruco_computed_virtual_to_opti[:,0]
        camera_calib_df["position_y"] = aruco_computed_virtual_to_opti[:,1]
        camera_calib_df["position_z"] = aruco_computed_virtual_to_opti[:,2]

        camera_calib_df = camera_calib_df[camera_calib_df["position_z"].apply(lambda x : x != 0)]

        camera_calib_df['time_delta'] = (camera_calib_df["converted_camera_timestamp"] - camera_first_timestamp).dt.total_seconds()

        camera_calib_df['2d_distance'] = np.sqrt(camera_calib_df['position_x'].astype(float)** 2 + camera_calib_df['position_z'].astype(float) ** 2)

        op_df = cleaned_opti_poses

        op_calib_df = op_df.iloc[:].copy()
        op_calib_df = op_calib_df.dropna()
        op_calib_df['2d_distance'] = np.sqrt(op_calib_df[position_x_key].astype(float)** 2 + op_calib_df[position_z_key].astype(float) ** 2)

        camera_pos = np.array(camera_calib_df['2d_distance']).astype(np.float32)
        camera_pos_zero_meaned = camera_pos - np.mean(camera_pos)
        camera_times = np.array(camera_calib_df['time_delta']).astype(np.float32)

        op_pos = np.array(op_calib_df['2d_distance']).astype(np.float32)
        op_times = np.array(op_calib_df['Time_Seconds']).astype(np.float32) # op's timestamp is in fact the time delta in op system

        op_interp = Akima1DInterpolator(op_times, op_pos)

        best_time, best_dist = -1, np.inf

        #basically, align camera_pos with op_pos at differnet start times by brute force 
        #this is rough alignment
        for i in range(len(op_times)):
            camera_times_shifted = camera_times + op_times[i]
            op_interped = op_interp(camera_times_shifted)
            op_interped_zero_meaned = op_interped - np.mean(op_interped)

            dist = np.mean(np.square(camera_pos_zero_meaned - op_interped_zero_meaned))

            if dist < best_dist:
                best_dist = dist
                best_time = i

        print("first pass")
        print(best_time, best_dist)

        camera_times_shifted = camera_times + op_times[best_time]

        #now, fine refinement with same technique
        #finding the best fit within 1 second of rough alignment
        best_time_refined, best_dist_refined = -1, np.inf
        for i in np.linspace(-0.5, 0.5, 1000):
            camera_times_refined = camera_times_shifted + i
            op_interped = op_interp(camera_times_refined)
            op_interped_zero_meaned = op_interped - np.mean(op_interped)

            dist = np.mean(np.square(camera_pos_zero_meaned - op_interped_zero_meaned))

            if dist < best_dist_refined:
                best_dist_refined = dist
                best_time_refined = i

        print("refinement pass")
        print(best_time_refined, best_dist_refined)

        camera_times_refined = camera_times_shifted + best_time_refined

        total_offset = op_times[best_time] + best_time_refined
        print("total offset to add to ak timestamps:", total_offset)

        camera_times_final = camera_times + total_offset
        op_interpd_final = op_interp(camera_times_final)
        op_interpd_final -= np.mean(op_interpd_final)

        if show_sync_plot:
            plt.scatter(camera_times_final, camera_pos_zero_meaned, label="az final")
            plt.scatter(camera_times_final, op_interpd_final, label="op final")
            plt.legend()
            plt.show()
            plt.clf()

        ### synchronizing poses using calculated offset ###

        camera_capture_df = camera_df.copy()
        camera_capture_df = camera_capture_df[camera_capture_df['Frame'].apply(lambda x : x >= capture_start_frame_id)]

        op_capture_df = op_df.copy()

        # calculate capture_time_off, sync opti-track to the clock of azure kinect
        capture_time_off = total_offset
        camera_capture_df['time_delta'] = (camera_capture_df['converted_camera_timestamp'] - camera_first_timestamp).dt.total_seconds() + capture_time_off # relate to the op start time
        camera_capture_times_synced_to_opti = np.array(camera_capture_df['time_delta']).astype(np.float32)

        op_capture_times = np.array(op_capture_df['Time_Seconds']).astype(np.float32)
        op_rotation_x_interp = Akima1DInterpolator(op_capture_times, op_capture_df[rotation_x_key])
        op_rotation_y_interp = Akima1DInterpolator(op_capture_times, op_capture_df[rotation_y_key])
        op_rotation_z_interp = Akima1DInterpolator(op_capture_times, op_capture_df[rotation_z_key])
        op_rotation_w_interp = Akima1DInterpolator(op_capture_times, op_capture_df[rotation_w_key])
        op_position_x_interp = Akima1DInterpolator(op_capture_times, op_capture_df[position_x_key])
        op_position_y_interp = Akima1DInterpolator(op_capture_times, op_capture_df[position_y_key])
        op_position_z_interp = Akima1DInterpolator(op_capture_times, op_capture_df[position_z_key])

        synced_df = camera_capture_df[["Frame"]].copy()

        synced_df['camera_Rotation_X'] = np.array(op_rotation_x_interp(camera_capture_times_synced_to_opti))
        synced_df['camera_Rotation_Y'] = np.array(op_rotation_y_interp(camera_capture_times_synced_to_opti))
        synced_df['camera_Rotation_Z'] = np.array(op_rotation_z_interp(camera_capture_times_synced_to_opti))
        synced_df['camera_Rotation_W'] = np.array(op_rotation_w_interp(camera_capture_times_synced_to_opti))
        synced_df['camera_Position_X'] = np.array(op_position_x_interp(camera_capture_times_synced_to_opti))
        synced_df['camera_Position_Y'] = np.array(op_position_y_interp(camera_capture_times_synced_to_opti))
        synced_df['camera_Position_Z'] = np.array(op_position_z_interp(camera_capture_times_synced_to_opti))

        synced_df.reset_index(inplace=True, drop=True)

        ### synchronization process ends ###

        output_frames_dir = os.path.join(scene_dir, "data")

        if not os.path.isdir(output_frames_dir):
            os.mkdir(output_frames_dir)

        output_raw_sync = os.path.join(scene_dir, "camera_poses", "camera_poses_synchronized_raw.csv")
        output_sync = os.path.join(scene_dir, "camera_poses", "camera_poses_synchronized.csv")\
            
        synced_df.to_csv(output_raw_sync)

        synced_df_renumbered = synced_df.copy()

        first_frame = synced_df_renumbered['Frame'].iloc[0]
        synced_df_renumbered["New_Frame"] = synced_df_renumbered['Frame'] - first_frame

        for _, row in synced_df_renumbered.iterrows():
            old_frame_id = int(row["Frame"])
            new_frame_id = int(row["New_Frame"])

            bgr = load_bgr(raw_frames_dir, old_frame_id)
            depth = load_depth(raw_frames_dir, old_frame_id)

            write_bgr(output_frames_dir, new_frame_id, bgr)
            write_depth(output_frames_dir, new_frame_id, depth)

        synced_df_renumbered["Frame"] = synced_df_renumbered["New_Frame"]
        synced_df_renumbered.drop("New_Frame", axis=1, inplace=True)

        synced_df_renumbered.to_csv(output_sync)

        #write scene metadata num_frames value
        scene_metadata["num_frames"] = synced_df_renumbered.shape[0]

        with open(scene_metadata_file, 'w') as file:
            yaml.dump(scene_metadata, file)

        return synced_df_renumbered
