"""
Cleans the exported tracking data output of the OptiTrack
"""

import cv2
import numpy as np
import struct
import threading
from tqdm import tqdm
import yaml

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from utils.camera_utils import write_frame_distortions, write_frame_intrinsics
from utils.constants import IPHONE_COLOR_WIDTH, IPHONE_COLOR_HEIGHT, IPHONE_DEPTH_WIDTH, IPHONE_DEPTH_HEIGHT
from utils.frame_utils import transfer_color_file, write_depth
from utils.pointcloud_utils import unproject_pixels

class IPhoneDataProcessor():
    def __init__(self):
        pass

    @staticmethod
    def compute_distorted_pt(lookup_table, distortion_optical_center, w, h, undistorted_idxs):
        delta_ocx_max = max(distortion_optical_center[0], w  - distortion_optical_center[0])
        delta_ocy_max = max(distortion_optical_center[1], h - distortion_optical_center[1])
        r_max = np.sqrt(delta_ocx_max * delta_ocx_max + delta_ocy_max * delta_ocy_max)

        v_point_x = undistorted_idxs[:,:,0] - distortion_optical_center[0]
        v_point_y = undistorted_idxs[:,:,1] - distortion_optical_center[1]
        
        r_point = np.sqrt(np.square(v_point_x) + np.square(v_point_y))

        val = np.clip(r_point, -np.inf, r_max) / r_max * (len(lookup_table)- 1)

        idx = val.astype(int)

        frac = val - idx

        mag = (1.0 - frac) * lookup_table[idx] + frac * lookup_table[np.clip(idx + 1, 0, len(lookup_table) - 1)]

        new_v_point_x = v_point_x + mag * v_point_x
        new_v_point_y = v_point_y + mag * v_point_y

        xs = new_v_point_x + distortion_optical_center[0]
        ys = new_v_point_y + distortion_optical_center[1]

        return xs, ys

    """
    pts_x = (N,)
    pts_y = (N,)
    """
    @staticmethod
    def bilinear_interp_color(img, pts_x, pts_y):
        pts_x_floored = np.floor(pts_x).astype(int)
        pts_x_ceil = np.ceil(pts_x).astype(int)

        pts_y_floored = np.floor(pts_y).astype(int)
        pts_y_ceil = np.ceil(pts_y).astype(int)

        pts_x_ratio = np.expand_dims(pts_x  - pts_x_floored, -1)
        pts_y_ratio = np.expand_dims(pts_y - pts_y_floored, -1)

        top_left_flattened = np.clip(pts_y_floored * img.shape[1] + pts_x_floored, 0, img.shape[0] * img.shape[1] - 1)
        top_right_flattened = np.clip(pts_y_floored * img.shape[1] + pts_x_ceil, 0, img.shape[0] * img.shape[1] - 1)
        bottom_left_flattened = np.clip(pts_y_ceil * img.shape[1] + pts_x_floored, 0, img.shape[0] * img.shape[1] - 1)
        bottom_right_flattened = np.clip(pts_y_ceil * img.shape[1] + pts_x_ceil, 0, img.shape[0] * img.shape[1] - 1)

        img_flattened = img.reshape((-1, 3))

        top_left_color = img_flattened[top_left_flattened].astype(np.float32)
        top_right_color = img_flattened[top_right_flattened].astype(np.float32)
        bottom_left_color = img_flattened[bottom_left_flattened].astype(np.float32)
        bottom_right_color = img_flattened[bottom_right_flattened].astype(np.float32)

        top_colors = top_left_color * (1 - pts_x_ratio) + top_right_color * pts_x_ratio
        bottom_colors = bottom_left_color * (1 - pts_x_ratio) + bottom_right_color * pts_x_ratio

        colors = top_colors * (1 - pts_y_ratio) + bottom_colors * pts_y_ratio

        colors[pts_x < 0] = np.zeros(3)
        colors[pts_x > img.shape[1] - 1] = np.zeros(3)
        colors[pts_y < 0] = np.zeros(3)
        colors[pts_y > img.shape[0] - 1] = np.zeros(3)
        colors = np.clip(colors, 0, 255).astype(np.uint8)

        return colors



    """
    Uses bilinear interpolation and inverse lookup to undistort RGB image
    """
    @staticmethod
    def undistort_color(img, lookup_table, distortion_optical_center):
        h, w, _ = img.shape

        x = np.arange(0, img.shape[1])
        y = np.arange(0, img.shape[0])

        # f_1 = interpolate.RectBivariateSpline(y, x, img[:,:,0])
        # f_2 = interpolate.RectBivariateSpline(y, x, img[:,:,1])
        # f_3 = interpolate.RectBivariateSpline(y, x, img[:,:,2])

        out = np.array([[[i, k] for i in range(img.shape[1])] for k in range(img.shape[0])])

        distorted_pts_x, distorted_pts_y = IPhoneDataProcessor.compute_distorted_pt(lookup_table, distortion_optical_center, w, h, out)

        distorted_pts_x = distorted_pts_x.flatten()
        distorted_pts_y = distorted_pts_y.flatten()

        out = IPhoneDataProcessor.bilinear_interp_color(img, distorted_pts_x, distorted_pts_y)
        out = out.reshape((h, w, 3))

        return out

    """
    Uses NN interpolation instead of bilinear interpolation since depth is non-linear
    """
    @staticmethod
    def undistort_depth(depth, lookup_table, distortion_optical_center):

        h, w = depth.shape

        out = np.array([[[i, k] for i in range(depth.shape[1])] for k in range(depth.shape[0])])

        distorted_pts_x, distorted_pts_y = IPhoneDataProcessor.compute_distorted_pt(lookup_table, distortion_optical_center, w, h, out)

        distorted_pts_x = np.round(distorted_pts_x.flatten()).astype(int)
        distorted_pts_y = np.round(distorted_pts_y.flatten()).astype(int)

        depth_flattened = depth.flatten()

        distorted_pts_idx = distorted_pts_y * w + distorted_pts_x

        distorted_pts_idx_clipped = np.clip(distorted_pts_idx, 0, h * w - 1)

        depth_undistorted_flattened = depth_flattened[distorted_pts_idx_clipped]
        depth_undistorted_flattened[distorted_pts_idx < 0] = 0
        depth_undistorted_flattened[distorted_pts_idx > h * w - 1] = 0

        depth_undistorted = depth_undistorted_flattened.reshape((h, w)).astype(np.uint16)

        return depth_undistorted

    @staticmethod
    def read_byte_float_file(file):
        arr = []
        with open(file, 'rb') as f:
            byte = f.read(4)

            while byte != b"":
                x = struct.unpack('<f', byte)
                arr.append(x[0])
                byte = f.read(4)

        return np.array(arr).astype(np.float32)

    @staticmethod
    def read_calib_file(file):
        with open(file, 'r') as f:
            _ = f.readline() # "Camera Intrinsic"
            arr = []
            for _ in range(3):
                line = f.readline().rstrip() 
                line = [float(x) for x in line.replace('[', '').replace(']', '').split(',') if len(x) > 0]
                arr.append(line)
            
            intr = np.array(arr).T

            _ = f.readline() # "Camera Extrinsic"
            arr = []
            for _ in range(4):
                line = f.readline().rstrip()
                line = [float(x) for x in line.replace('[', '').replace(']', '').split(',') if len(x) > 0]
                arr.append(line)
            
            # Extrinsic should be Identity
            extr = np.array(arr).T
            assert(np.array_equal(extr[:,:3], np.eye(3)))
            assert(np.array_equal(extr[:,3], np.zeros(3)))

            _ = f.readline() # "Distortion center"
            dist_center = [float(x) for x in f.readline().rstrip().split(',') if len(x) > 0]

            return intr, dist_center

    @staticmethod
    def compute_distortion_coeffs(depth, lookup_table, distortion_center, intr):
        # Test distortion parameter fitting
        num_sampled_points = 20000

        y_sampled = np.round(np.arange(0, IPHONE_COLOR_HEIGHT, IPHONE_COLOR_HEIGHT / num_sampled_points))
        x_sampled = np.round(np.arange(0, IPHONE_COLOR_WIDTH, IPHONE_COLOR_WIDTH / num_sampled_points))

        np.random.shuffle(y_sampled)
        np.random.shuffle(x_sampled)

        y_sampled = np.expand_dims(y_sampled, 1)
        x_sampled = np.expand_dims(x_sampled, 1)

        points_sampled = np.concatenate((x_sampled, y_sampled), 1)

        h, w = depth.shape

        undistorted_idxs = np.expand_dims(points_sampled, 0)

        distorted_pts_x, distorted_pts_y = IPhoneDataProcessor.compute_distorted_pt(lookup_table, distortion_center, w, h, undistorted_idxs)

        distorted_pts_x = distorted_pts_x.squeeze()
        distorted_pts_y = distorted_pts_y.squeeze()

        distorted_pts = np.concatenate((np.expand_dims(distorted_pts_x, 1), np.expand_dims(distorted_pts_y, 1)), 1)

        flattened_sampled = np.round(np.round(distorted_pts_y) * IPHONE_COLOR_WIDTH + distorted_pts_x).astype(int)
        flattened_sampled_clipped = np.clip(flattened_sampled, 0 , h * w - 1)

        depths_sampled = depth.flatten()[flattened_sampled_clipped]
        depths_sampled[flattened_sampled < 0] = 0
        depths_sampled[flattened_sampled > h * w - 1] = 0

        points_sampled_3d = unproject_pixels(points_sampled, depths_sampled, 0.001, intr, None)

        mask = points_sampled_3d[:,2] > 0

        distorted_pts = distorted_pts[mask]
        points_sampled = points_sampled[mask]
        points_sampled_3d = points_sampled_3d[mask]

        points_sampled_3d = np.expand_dims(points_sampled_3d.astype(np.float32), 0)
        distorted_pts = np.expand_dims(distorted_pts.astype(np.float32), 0)

        flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_PRINCIPAL_POINT
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(points_sampled_3d, distorted_pts, (w, h), np.ascontiguousarray(intr.astype(np.float32)), None, flags=flags)

        return dist

    @staticmethod
    def process_iphone_frames(frame_ids, iphone_data_input, data_raw_output, intrs, dists):
        for frame_id in tqdm(frame_ids, total=len(frame_ids), desc="collecting intrinsics/distortions and moving frames"):
            lookup_table_file = os.path.join(iphone_data_input, "{0}_distortion_table.bin".format(frame_id))
            calib_file = os.path.join(iphone_data_input, "{0}_calibration.txt".format(frame_id))
            color_file = os.path.join(iphone_data_input, "{0}.jpeg".format(frame_id))

            lookup_table = IPhoneDataProcessor.read_byte_float_file(lookup_table_file)
            intr, distortion_center = IPhoneDataProcessor.read_calib_file(calib_file)

            depth_old = os.path.join(iphone_data_input, "{0}.bin".format(frame_id))
            depth_arr = []

            with open(depth_old, 'rb') as f:
                byte = f.read(2)

                while byte != b"":
                    x = int.from_bytes(byte, "little", signed=False)
                    depth_arr.append(x)
                    byte = f.read(2)

            depth = np.array(depth_arr).astype(np.uint16)

            # Depth data is sideways
            depth = depth.reshape((IPHONE_DEPTH_HEIGHT, IPHONE_DEPTH_WIDTH))

            # Interpolate nearest for depth -> color size
            depth = cv2.resize(depth, (IPHONE_COLOR_WIDTH, IPHONE_COLOR_HEIGHT), cv2.INTER_NEAREST)

            distortion_coeffs = IPhoneDataProcessor.compute_distortion_coeffs(depth, lookup_table, distortion_center, intr)

            intrs[frame_id] = intr
            dists[frame_id] = distortion_coeffs

            transfer_color_file(color_file, data_raw_output, frame_id, "jpg")
            write_depth(data_raw_output, frame_id, depth)

    @staticmethod
    def process_iphone_scene_data(scene_dir, camera_name):

        iphone_data_input = os.path.join(scene_dir, "iphone_data")
        assert(os.path.isdir(iphone_data_input))

        frames = os.listdir(iphone_data_input)
        frame_ids = [f[:f.find(".")] for f in frames if ".jpeg" in f]
        frame_ids = sorted([int(f) for f in frame_ids])

        timestamp_data_file = os.path.join(scene_dir, "timestamps.csv")
        with open(timestamp_data_file, "r") as f:
            timestamp_data = f.readline()

        timestamp_data = timestamp_data.rstrip().split(",")

        frame_timestamps = [float(t) for t in timestamp_data[:-1]]
        capture_start_frame = int(timestamp_data[-1])

        print("number of frames:", len(frame_ids))

        assert(len(frame_timestamps) == len(frame_ids))

        #output files
        data_raw_output = os.path.join(scene_dir, "data_raw")

        if not os.path.isdir(data_raw_output):
            os.mkdir(data_raw_output)

        camera_data_output = open(os.path.join(scene_dir, "camera_data.csv"), 'w')
        camera_time_break_output = open(os.path.join(scene_dir, "camera_time_break.csv"), 'w')
        scene_metadata_output = os.path.join(scene_dir, "scene_meta.yaml")

        camera_data_output.write("Frame,Timestamp\n")
        camera_time_break_output.write("Calibration Start ID,Calibration End ID,Capture Start ID\n")

        for frame_id, frame_timestamp in zip(frame_ids, frame_timestamps):
            camera_data_output.write("{0},{1}\n".format(frame_id, frame_timestamp))

        camera_time_break_output.write("{0},{1},{2}\n".format(0, capture_start_frame, capture_start_frame))

        camera_data_output.close()
        camera_time_break_output.close()

        intrs = {}
        dists = {}

        thread_count = 10
        threads = []

        for i in range(thread_count):
            threads.append(threading.Thread(target=IPhoneDataProcessor.process_iphone_frames, args=(frame_ids[i::thread_count], iphone_data_input, data_raw_output, intrs, dists)))
        for i in range(thread_count):
            threads[i].start()
        for i in range(thread_count):
            threads[i].join()

        print("intrs:", len(intrs))
        print("dists:", len(dists))

        write_frame_intrinsics(camera_name, scene_dir, intrs, raw=True)
        write_frame_distortions(camera_name, scene_dir, dists, raw=True)

        scene_metadata = {}
        scene_metadata["cam_scale"] = 0.001
        scene_metadata["camera"] = camera_name
        scene_metadata["objects"] = []

        with open(scene_metadata_output, "w") as f:
            yaml.dump(scene_metadata, f)

        print("Transfered data into Azure Kinect format.")
        print("PLEASE fill the objects field in the scene_metadata.yaml")
