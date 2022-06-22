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

def write_archive_extrinsic(extrinsics, scene_dir):
    archive_extrinsic_path = os.path.join(scene_dir, "annotated_object_poses", "archive_extrinsic.txt")
    np.savetxt(archive_extrinsic_path, extrinsics)

def write_extrinsics(camera_name, extrinsics):
    camera_path = os.path.join(cameras_dir, camera_name)
    assert(os.path.isdir(camera_path))
    extrinsic_path = os.path.join(camera_path, "extrinsic.txt")
    np.savetxt(extrinsic_path, extrinsics)

def load_extrinsics(camera_name, scene_dir=None, use_archive=True):
    if scene_dir and use_archive:
        archive_extrinsic_path = os.path.join(scene_dir, "annotated_object_poses", "archive_extrinsic.txt")
        if os.path.isfile(archive_extrinsic_path):
            return np.loadtxt(archive_extrinsic_path)
        else:
            print("Use_archive set to True, but no archived extrinsic found. Using camera extrinsic.")
    camera_path = os.path.join(cameras_dir, camera_name)
    assert(os.path.isdir(camera_path))
    extrinsic_path = os.path.join(camera_path, "extrinsic.txt")
    print("Loading camera {0} default extrinsic.".format(camera_name))
    return np.loadtxt(extrinsic_path)
