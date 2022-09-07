import cv2
import json
import numpy as np
import open3d as o3d
from PIL import Image
import shutil

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

def validate_extension(ext):
    if ext not in ["jpeg", "jpg", "png"]:
        raise "Invalid Extension! {0}".format(ext)

def extension_match(ext1, ext2):
    if ext1 == ext2:
        return True
    if ext1 in ["jpeg", "jpg"] and ext2 in ["jpeg", "jpg"]:
        return True
    return False

def write_bgr(frames_dir, frame_id, frame, ext):
    validate_extension(ext)
    frame_name = os.path.join(frames_dir, str(frame_id).zfill(5) + "_color.{0}".format(ext))
    cv2.imwrite(frame_name, frame)

def load_bgr(frames_dir, frame_id, ext):
    validate_extension(ext)
    frame = cv2.imread(os.path.join(frames_dir, str(frame_id).zfill(5) + "_color.{0}".format(ext)), cv2.IMREAD_COLOR)
    return frame

def write_rgb(frames_dir, frame_id, frame, ext):
    validate_extension(ext)
    frame = Image.fromarray(frame)
    frame_name = os.path.join(frames_dir, str(frame_id).zfill(5) + "_color.{0}".format(ext))
    frame.save(frame_name)

def write_debug_rgb(frames_dir, frame_id, frame, ext):
    validate_extension(ext)
    frame = Image.fromarray(frame)
    frame_name = os.path.join(frames_dir, str(frame_id).zfill(5) + "_color_debug.{0}".format(ext))
    frame.save(frame_name)

def load_rgb(frames_dir, frame_id, ext):
    validate_extension(ext)
    frame = np.array(Image.open(os.path.join(frames_dir, str(frame_id).zfill(5) + "_color.{0}".format(ext))))
    return frame

def transfer_color(old_frames_dir, old_frame_id, old_ext, new_frames_dir, new_frame_id, new_ext):
    validate_extension(old_ext)
    validate_extension(new_ext)
    old_frame_name = os.path.join(old_frames_dir, str(old_frame_id).zfill(5) + "_color.{0}".format(old_ext))
    new_frame_name = os.path.join(new_frames_dir, str(new_frame_id).zfill(5) + "_color.{0}".format(new_ext))
    if extension_match(old_ext, new_ext):
        shutil.copyfile(old_frame_name, new_frame_name)
    else:
        img = cv2.imread(old_frame_name, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(new_frame_name, img)

def transfer_color_file(color_file, new_frames_dir, new_frame_id, new_ext):
    old_ext = color_file[color_file.rfind(".") + 1:]
    validate_extension(old_ext)
    validate_extension(new_ext)
    new_frame_name = os.path.join(new_frames_dir, str(new_frame_id).zfill(5) + "_color.{0}".format(new_ext))
    if extension_match(old_ext, new_ext):
        shutil.copyfile(color_file, new_frame_name)
    else:
        img = cv2.imread(color_file, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(new_frame_name, img)

def get_color_ext(frames_dir):
    frames = os.listdir(frames_dir)
    color_frames = [f for f in frames if "color" in f]
    color_exts = [f[f.rfind(".") + 1:] for f in color_frames]
    if color_exts.count(color_exts[0]) != len(color_exts):
        raise("Not all same extension!")
    validate_extension(color_exts[0])
    return color_exts[0]

def write_depth(frames_dir, frame_id, frame):
    assert(frame.dtype == np.uint16)
    frame_name = os.path.join(frames_dir, str(frame_id).zfill(5) + "_depth.png")
    cv2.imwrite(frame_name, frame)

def load_depth(frames_dir, frame_id):
    frame = cv2.imread(os.path.join(frames_dir, str(frame_id).zfill(5) + "_depth.png"), cv2.IMREAD_UNCHANGED)
    return frame

def transfer_depth(old_frames_dir, old_frame_id, new_frames_dir, new_frame_id):
    old_frame_name = os.path.join(old_frames_dir, str(old_frame_id).zfill(5) + "_depth.png")
    new_frame_name = os.path.join(new_frames_dir, str(new_frame_id).zfill(5) + "_depth.png")
    shutil.copyfile(old_frame_name, new_frame_name)

def load_o3d_rgb(frames_dir, frame_id, ext):
    validate_extension(ext)
    path = os.path.join(frames_dir, str(frame_id).zfill(5) + "_color.{0}".format(ext))
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
        if len(ids) > 1:
            print("Warning, multiple ARUCO's detected. Returning None.")
            return None
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

        return (rvec, tvec, corners)

    else:
        return None
