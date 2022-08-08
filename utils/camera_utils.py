from collections import defaultdict
import numpy as np
import os
import yaml

dir_path = os.path.dirname(os.path.realpath(__file__))
cameras_dir = os.path.join(dir_path, "..", "cameras")

"""
Behavior for intrinsics and distortions:

Intrinsics for each scene are stored in intrinsic.yaml and intrinsic_raw.yaml / distortion.yaml and distortion_raw.yaml.

Each yaml file contains either 1 of 2 keys: 1) static_intrinsic 2) frame_intrinsics / 1) static_distortion 2) frame_distortions.

If [static_intrinsic] exists, the same intrinsic is used for every frame in the scene.
If [frame_intrinsics] exists, the intrinsic is different for each frame, indexed by [frame_id].

/

If [static_distortion] exists, the same distortion is used for every frame in the scene.
If [frame_distortions] exists, the distortion is different for each frame, idnexed by [frame_id].

load_frame_intrinsics/load_frame_distortions always returns a dictionary mapping [frame_id] to intrinsic matrix.
For scenes with a static intrinsic, this is implemented as a defaultdict.
"""


def load_intrinsic_static(camera_name):
    if "iphone" in camera_name:
        raise "iPhone doesn't have a static intrinsic."
    camera_path = os.path.join(cameras_dir, camera_name)
    assert(os.path.isdir(camera_path))
    intrinsics_path = os.path.join(camera_path, "intrinsic.txt")
    return np.loadtxt(intrinsics_path)

def load_distortion_static(camera_name):
    if "az" in camera_name:
        camera_path = os.path.join(cameras_dir, camera_name)
        assert(os.path.isdir(camera_path))
        distortion_path = os.path.join(camera_path, "distortion.txt")
        return np.loadtxt(distortion_path)
    # iPhone images are already undistorted upon capture
    elif "iphone" in camera_name:
        return None
    else:
        raise "Unsupported camera {0}".format(camera_name)

def write_static_intrinsic(camera_name, scene_dir, raw):
    if "iphone" in camera_name:
        raise "Use [write_frame_intrinsics] for iPhone scenes."

    if raw:
        intrinsic_file = os.path.join(scene_dir, "intrinsic_raw.yaml")
    else:
        intrinsic_file = os.path.join(scene_dir, "intrinsic.yaml")

    intrinsic = load_intrinsic_static(camera_name)
    intrinsic = intrinsic.tolist()

    intrinsic_out = {}
    intrinsic_out["static_intrinsic"] = intrinsic

    with open(intrinsic_file, 'w') as outfile:
        yaml.dump(intrinsic_out, outfile)

def write_static_distortion(camera_name, scene_dir, raw):
    if "iphone" in camera_name:
        raise "Use [write_frame_distortions] for iPhone scenes."

    if raw:
        distortion_file = os.path.join(scene_dir, "distortion_raw.yaml")
    else:
        distortion_file = os.path.join(scene_dir, "distortion.yaml")

    distortion = load_distortion_static(camera_name)
    distortion = distortion.tolist()

    distortion_out = {}
    distortion_out["static_distortion"] = distortion

    with open(distortion_file, 'w') as outfile:
        yaml.dump(distortion_out, outfile)

def write_frame_intrinsics(camera_name, scene_dir, frame_intrinsics, raw):
    if "az" in camera_name:
        raise "Use [write_static_intrinsic] for Azure Kinect scenes."
    
    frame_intrinsics = {k: v.tolist() for k, v in frame_intrinsics.items()}

    if raw:
        intrinsic_file = os.path.join(scene_dir, "intrinsic_raw.yaml")
    else:
        intrinsic_file = os.path.join(scene_dir, "intrinsic.yaml")

    intrinsic_out = {}
    intrinsic_out["frame_intrinsics"] = frame_intrinsics

    with open(intrinsic_file, 'w') as outfile:
        yaml.dump(intrinsic_out, outfile)

def write_frame_distortions(camera_name, scene_dir, frame_distortions, raw):
    if "az" in camera_name:
        raise "Use [write_static_distortion] for Azure Kinect scenes."

    frame_distortions = {k: v.tolist() for k, v in frame_distortions.items()}

    if raw:
        distortion_file = os.path.join(scene_dir, "distortion_raw.yaml")
    else:
        distortion_file = os.path.join(scene_dir, "distortion.yaml")

    distortion_out = {}
    distortion_out["frame_distortions"] = frame_distortions

    with open(distortion_file, 'w') as outfile:
        yaml.dump(distortion_out, outfile)

def write_scene_intrinsics(camera_name, scene_dir, frame_intrinsics, raw):
    if "az" in camera_name:
        write_static_intrinsic(camera_name, scene_dir, raw)
    elif "iphone" in camera_name:
        write_frame_intrinsics(camera_name, scene_dir, frame_intrinsics, raw)
    else:
        raise "Unknown camera type {0}".format(camera_name)

def write_scene_distortions(camera_name, scene_dir, frame_distortions, raw):
    if "az" in camera_name:
        write_static_distortion(camera_name, scene_dir, raw)
    elif "iphone" in camera_name:
        write_frame_distortions(camera_name, scene_dir, frame_distortions, raw)
    else:
        raise "Unknown camera type {0}".format(camera_name)

def load_frame_intrinsics(scene_dir, raw):
    if raw:
        intrinsic_file = os.path.join(scene_dir, "intrinsic_raw.yaml")
    else:
        intrinsic_file = os.path.join(scene_dir, "intrinsic.yaml")

    with open(intrinsic_file, "r") as file:
        intrinsics = yaml.safe_load(file)

    if "static_intrinsic" in intrinsics.keys():
        intrinsic = np.array(intrinsics["static_intrinsic"])
        return defaultdict(lambda: intrinsic)
    else:
        frame_intrinsics = intrinsics["frame_intrinsics"]
        frame_intrinsics = {k: np.array(v) for k, v in frame_intrinsics.items()}
        return frame_intrinsics

def load_frame_distortions(scene_dir, raw):
    if raw:
        distortion_file = os.path.join(scene_dir, "distortion_raw.yaml")
    else:
        distortion_file = os.path.join(scene_dir, "distortion.yaml")

    with open(distortion_file, 'r') as file:
        distortions = yaml.safe_load(file)
    
    if "static_distortion" in distortions.keys():
        distortion = np.array(distortions["static_distortion"])
        return defaultdict(lambda: distortion)
    else:
        frame_distortions = distortions["frame_distortions"]
        frame_distortions = {k: np.array(v) for k, v in frame_distortions.items()}
        return frame_distortions

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
            print("Using archived extrinsic.")
            return np.loadtxt(archive_extrinsic_path)
        else:
            print("Use_archive set to True, but no archived extrinsic found. Using camera extrinsic.")
    camera_path = os.path.join(cameras_dir, camera_name)
    assert(os.path.isdir(camera_path))
    extrinsic_path = os.path.join(camera_path, "extrinsic.txt")
    print("Loading camera {0} default extrinsic.".format(camera_name))
    return np.loadtxt(extrinsic_path)
