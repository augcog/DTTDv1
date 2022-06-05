"""

Eventually, this code will be some sort of simulator using the poses of the camera and the annotated/recovered poses
of the objects. This will just be used to verify the quality of the dataset.

Actually, this code doesn't feel that useful...

"""

import argparse
from re import S
import cv2
import os
import yaml

from utils.frame_utils import load_bgr, load_meta
from utils.object_utils import load_object_meshes

def main():
    parser = argparse.ArgumentParser(description='Verification of the dataset!')
    parser.add_argument('scene_dir', type=str, help='scene directory (contains scene_meta.yaml and data (frames) and camera_poses)')

    args = parser.parse_args()

    scene_metadata_file = os.path.join(args.scene_dir, "scene_meta.yaml")
    with open(scene_metadata_file, 'r') as file:
        scene_metadata = yaml.safe_load(file)
    
    objects = load_object_meshes(scene_metadata["objects"])

    frames_dir = os.path.join(args.scene_dir, "data")

    cv2.namedWindow("frame")

    for frame_id in range(scene_metadata["num_frames"]):
        bgr = load_bgr(frames_dir, frame_id)
        meta = load_meta(frames_dir, frame_id)

        cv2.imshow("frame", bgr)
        cv2.waitKey(30)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()