import os
import cv2
from PIL import Image
import numpy as np

def load_bgr(frames_dir, frame_id):
    frame = cv2.imread(os.path.join(frames_dir, str(frame_id).zfill(5) + "_color.jpg"), cv2.IMREAD_COLOR)
    return frame

def load_rgb(frames_dir, frame_id):
    frame = np.array(Image.open(os.path.join(frames_dir, str(frame_id).zfill(5) + "_color.jpg")))
    return frame

def load_depth(frames_dir, frame_id):
    frame = cv2.imread(os.path.join(frames_dir, str(frame_id).zfill(5) + "_depth.png"), cv2.IMREAD_UNCHANGED)
    return frame

def write_label(frames_dir, frame_id, frame):
    frame_name = os.path.join(frames_dir, str(frame_id).zfill(5) + "_label.png")
    cv2.imwrite(frame_name, frame)