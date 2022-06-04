"""

Eventually, this code will be some sort of simulator using the poses of the camera and the annotated/recovered poses
of the objects. This will just be used to verify the quality of the dataset.

"""

import argparse

def main():
    parser = argparse.ArgumentParser(description='Verification of the dataset!')
    parser.add_argument('scene_dir', type=str, help='scene directory (contains scene_meta.yaml and data (frames) and camera_poses)')

if __name__ == "__main__":
    main()