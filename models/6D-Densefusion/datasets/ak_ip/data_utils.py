import numpy as np
import os
import pandas as pd
import yaml

def load_objects_dir(objects_dir):

    object_ids_file = os.path.join(objects_dir, "objectids.csv")
    object_ids_df = pd.read_csv(object_ids_file)

    object_pclds = {}

    for _, row in object_ids_df.iterrows():
        object_id, object_name = row[0], row[1]
        points_path = os.path.join(objects_dir, object_name, "points.xyz")
        input_file = open(points_path, 'r')

        points = []
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            input_line = input_line[:-1].split(' ')
            points.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
        
        input_file.close()

        object_pclds[object_id] = {'name': object_name, 'pcld': np.array(points)}

    return object_pclds

def is_camera(f):
    if not os.path.isdir(f):
        return False
    
    if not os.path.isfile(os.path.join(f, "intrinsic.txt")):
        return False

    if not os.path.isfile(os.path.join(f, "distortion.txt")):
        return False

    return True

def load_cameras_dir(cameras_dir):

    cameras = {}

    for cam in os.listdir(cameras_dir):
        cam_abs = os.path.join(cameras_dir, cam)
        if is_camera(cam_abs):
            intrinsic_path = os.path.join(cam_abs, "intrinsic.txt")
            distortion_path = os.path.join(cam_abs, "distortion.txt")
            intrinsic = np.loadtxt(intrinsic_path)
            distortion = np.loadtxt(distortion_path)

            cameras[cam] = {}
            cameras[cam]["intrinsic"] = intrinsic
            cameras[cam]["distortion"] = distortion

    return cameras

def load_data_list(data_list_file):
    data_list = []

    with open(data_list_file, "r") as f:
        for line in f:
            data_list.append(line.rstrip())

    return data_list

def load_scene_metas(data_dir):

    scene_metadatas = {}

    for scene in os.listdir(data_dir):
        scene_abs = os.path.join(data_dir, scene)
        if not os.path.isdir(scene_abs):
            continue

        scene_metadata_file = os.path.join(scene_abs, "scene_meta.yaml")

        with open(scene_metadata_file, "r") as f:
            scene_metadata = yaml.safe_load(f)

        scene_metadatas[scene] = scene_metadata

    return scene_metadatas