import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
cameras_dir = os.path.join(dir_path, "..", "cameras")

def load_intrinsics(camera_name):
    camera_path = os.path.join(cameras_dir, camera_name)
    assert(os.path.isdir(camera_path))
    intrinsics_path = os.path.join(camera_path, "intrinsic.txt")
    return np.loadtxt(intrinsics_path)

def load_distortion(camera_name):
    camera_path = os.path.join(cameras_dir, camera_name)
    assert(os.path.isdir(camera_path))
    distortion_path = os.path.join(camera_path, "distortion.txt")
    return np.loadtxt(distortion_path)

def write_extrinsics(camera_name, extrinsics):
    camera_path = os.path.join(cameras_dir, camera_name)
    assert(os.path.isdir(camera_path))
    extrinsic_path = os.path.join(camera_path, "extrinsic.txt")
    np.savetxt(extrinsic_path, extrinsics)

def load_extrinsics(camera_name):
    camera_path = os.path.join(cameras_dir, camera_name)
    assert(os.path.isdir(camera_path))
    extrinsic_path = os.path.join(camera_path, "extrinsic.txt")
    return np.loadtxt(extrinsic_path)
