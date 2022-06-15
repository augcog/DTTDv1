import cv2
import json
import numpy as np
import open3d as o3d
import os
from PIL import Image

def write_bgr(frames_dir, frame_id, frame):
    frame_name = os.path.join(frames_dir, str(frame_id).zfill(5) + "_color.jpg")
    cv2.imwrite(frame_name, frame)

def load_bgr(frames_dir, frame_id):
    frame = cv2.imread(os.path.join(frames_dir, str(frame_id).zfill(5) + "_color.jpg"), cv2.IMREAD_COLOR)
    return frame

def write_rgb(frames_dir, frame_id, frame):
    frame = Image.fromarray(frame)
    frame_name = os.path.join(frames_dir, str(frame_id).zfill(5) + "_color.jpg")
    frame.save(frame_name)

def write_debug_rgb(frames_dir, frame_id, frame):
    frame = Image.fromarray(frame)
    frame_name = os.path.join(frames_dir, str(frame_id).zfill(5) + "_color_debug.jpg")
    frame.save(frame_name)

def load_rgb(frames_dir, frame_id):
    frame = np.array(Image.open(os.path.join(frames_dir, str(frame_id).zfill(5) + "_color.jpg")))
    return frame

def write_depth(frames_dir, frame_id, frame):
    assert(frame.dtype == np.uint16)
    frame_name = os.path.join(frames_dir, str(frame_id).zfill(5) + "_depth.png")
    cv2.imwrite(frame_name, frame)

def load_depth(frames_dir, frame_id):
    frame = cv2.imread(os.path.join(frames_dir, str(frame_id).zfill(5) + "_depth.png"), cv2.IMREAD_UNCHANGED)
    return frame

def load_o3d_rgb(frames_dir, frame_id):
    path = os.path.join(frames_dir, str(frame_id).zfill(5) + "_color.jpg")
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
