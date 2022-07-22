"""

Tool to review annotations for a scene.

"""

import argparse
import yaml

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from utils.constants import SCENES_DIR
from utils.object_utils import load_object_meshes_trimesh
from quality_control import AnnotationReviewer

def main():
    parser = argparse.ArgumentParser(description='Generate semantic labeling and meta labeling')
    parser.add_argument('scene_name', type=str, help='scene directory (contains scene_meta.yaml and data (frames) and camera_poses)')
    args = parser.parse_args()

    scene_dir = os.path.join(SCENES_DIR, args.scene_name)

    scene_metadata_file = os.path.join(scene_dir, "scene_meta.yaml")
    with open(scene_metadata_file, 'r') as file:
        scene_metadata = yaml.safe_load(file)
    
    objects = load_object_meshes_trimesh(scene_metadata["objects"])

    annotation_reviewer = AnnotationReviewer(objects)

    annotation_reviewer.review_scene_annotations(scene_dir)

if __name__ == "__main__":
    main()


    