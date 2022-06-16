import cv2
import json
import numpy as np
import open3d as o3d
import os
from PIL import Image

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from utils.pointcloud_utils import unproject_pixels

def write_bgr(frames_dir, frame_id, frame):
    frame_name = os.path.join(frames_dir, str(frame_id).zfill(5) + "_color.png")
    cv2.imwrite(frame_name, frame)

def load_bgr(frames_dir, frame_id):
    frame = cv2.imread(os.path.join(frames_dir, str(frame_id).zfill(5) + "_color.png"), cv2.IMREAD_COLOR)
    return frame

def write_rgb(frames_dir, frame_id, frame):
    frame = Image.fromarray(frame)
    frame_name = os.path.join(frames_dir, str(frame_id).zfill(5) + "_color.png")
    frame.save(frame_name)

def write_debug_rgb(frames_dir, frame_id, frame):
    frame = Image.fromarray(frame)
    frame_name = os.path.join(frames_dir, str(frame_id).zfill(5) + "_color_debug.png")
    frame.save(frame_name)

def load_rgb(frames_dir, frame_id):
    frame = np.array(Image.open(os.path.join(frames_dir, str(frame_id).zfill(5) + "_color.png")))
    return frame

def write_depth(frames_dir, frame_id, frame):
    assert(frame.dtype == np.uint16)
    frame_name = os.path.join(frames_dir, str(frame_id).zfill(5) + "_depth.png")
    cv2.imwrite(frame_name, frame)

def load_depth(frames_dir, frame_id):
    frame = cv2.imread(os.path.join(frames_dir, str(frame_id).zfill(5) + "_depth.png"), cv2.IMREAD_UNCHANGED)
    return frame

def load_o3d_rgb(frames_dir, frame_id):
    path = os.path.join(frames_dir, str(frame_id).zfill(5) + "_color.png")
    frame = o3d.io.read_image(path)
    return frame

def load_o3d_depth(frames_dir, frame_id):
    path = os.path.join(frames_dir, str(frame_id).zfill(5) + "_depth.png")
    frame = o3d.io.read_image(path)
    return frame

def write_label(frames_dir, frame_id, frame):
    frame_name = os.path.join(frames_dir, str(frame_id).zfill(5) + "_label.png")
    cv2.imwrite(frame_name, frame)

def write_debug_label(frames_dir, frame_id, frame):
    frame_name = os.path.join(frames_dir, str(frame_id).zfill(5) + "_label_debug.png")
    cv2.imwrite(frame_name, frame)

def load_label(frames_dir, frame_id):
    frame = cv2.imread(os.path.join(frames_dir, str(frame_id).zfill(5) + "_label.png"), cv2.IMREAD_UNCHANGED)
    return frame

def write_meta(frames_dir, frame_id, meta):
    meta_file = os.path.join(frames_dir, str(frame_id).zfill(5) + "_meta.json")
    with open(meta_file, "w") as f:
        json.dump(meta, f)

def load_meta(frames_dir, frame_id):
    meta_file = os.path.join(frames_dir, str(frame_id).zfill(5) + "_meta.json")
    with open(meta_file, "r") as f:
        meta = json.load(f)
    return meta

def calculate_aruco_from_bgr_and_depth(bgr, depth, depth_scale, camera_matrix, camera_dist, aruco_dictionary, aruco_parameters):
    # Detect the markers in the image
    corners, ids, _ = cv2.aruco.detectMarkers(bgr, aruco_dictionary, parameters=aruco_parameters)

    if np.all(ids is not None):  # If there are markers found by detector
        assert(len(ids) == 1)
        # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[0], 0.02, camera_matrix,
                                                                    camera_dist)
        #1x3
        rvec = np.squeeze(rvec, axis = 1)
        #1x3
        tvec = np.squeeze(tvec, axis = 1) * 9

        #refine with Azure Kinect depth
        center_projected, _ = cv2.projectPoints(tvec, np.array([[0, 0, 0]]).astype(np.float32), np.array([0, 0, 0]).astype(np.float32), camera_matrix, camera_dist)
        center_projected = center_projected.squeeze()
        center_x, center_y = center_projected

        center_x, center_y = int(center_x), int(center_y)
        center_depth = np.array([depth[center_y, center_x] * depth_scale])

        center_pt = tvec / tvec[0, 2] * center_depth

        tvec = center_pt

        return rvec, tvec, corners

    else:
        return None