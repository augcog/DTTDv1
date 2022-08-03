"""
Generate actual dataset output.
"""

import argparse
import random
import shutil
import numpy as np

import os, sys

from sklearn.model_selection import train_test_split 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

ROOT_TRAIN_DATA_LIST_FILE = "train_data_list.txt"
ROOT_TEST_DATA_LIST_FILE = "test_data_list.txt"
SKIP_FRAMES_NUM = 1 # Number of frames skipped between each frame selected
TRAIN_TEST_SPLIT_PERCENTAGE = 0.9

def main():
    
    #SET CONSISTENT SEED
    random.seed(2022)
    np.random.seed(2022)

    parser = argparse.ArgumentParser(description='Generate data root')
    parser.add_argument('scene_list_file', type=str, help="File containing a list of scene names, one per line, to be gathered into the dataset.")
    parser.add_argument('--scene_dir', type=str, default=os.path.join(dir_path, "..", "scenes"), help="Directory containing scenes to be gathered")
    parser.add_argument('--output', type=str, default=os.path.join(dir_path, "..", "toolbox", "root"))
    parser.add_argument('--move_cameras', action="store_true")
    parser.add_argument('--move_objects', action="store_true")
    parser.add_argument('--move_scenes', action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    else:
        print("Warning!! Output folder {0} already exists.".format(args.output))

    scenes = []

    with open(args.scene_list_file, "r") as f:
        for line in f:
            line = line.rstrip()
            scenes.append(line)

    scene_path_list = [os.path.join(args.scene_dir, scene_name) for scene_name in scenes]
    for scene in scene_path_list:
        assert(os.path.isdir(scene))

    random.shuffle(scene_path_list)
    train_folder_path_list = scene_path_list[:int(len(scene_path_list) * TRAIN_TEST_SPLIT_PERCENTAGE)]
    test_folder_path_list = scene_path_list[int(len(scene_path_list) * TRAIN_TEST_SPLIT_PERCENTAGE):]

    print("Number of training scenes: {0}. Number of testing scenes: {1}".format(len(train_folder_path_list), len(test_folder_path_list)))

    with open(os.path.join(args.output, ROOT_TRAIN_DATA_LIST_FILE), 'w') as f:
        for each_scene_path in train_folder_path_list:
            data_dir = os.path.join(each_scene_path, "data")
            upper_limits = len(os.listdir(data_dir)) // 5
            ids = [str(x).zfill(5) for x in list(range(0, upper_limits, SKIP_FRAMES_NUM))]
            file_names = [(str(os.path.basename(os.path.normpath(each_scene_path))) + "/data/" + each_id +'\n') for each_id in ids]
            f.writelines(file_names)

    with open(os.path.join(args.output, ROOT_TEST_DATA_LIST_FILE), "w") as f:
        for each_scene_path in test_folder_path_list:
            data_dir = os.path.join(each_scene_path, "data")
            upper_limits = len(os.listdir(data_dir)) // 5
            ids = [str(x).zfill(5) for x in list(range(0, upper_limits, SKIP_FRAMES_NUM))]
            file_names = [(str(os.path.basename(os.path.normpath(each_scene_path))) + "/data/" + each_id + "\n") for each_id in ids]
            f.writelines(file_names) 

    if args.move_cameras:
        print("Moving cameras")
        cameras_input = os.path.join(dir_path, "..", "cameras")
        cameras_output = os.path.join(args.output, "cameras")
        shutil.copytree(cameras_input, cameras_output)

    if args.move_objects:
        print("Moving objects")
        object_input = os.path.join(dir_path, "..", "objects")
        object_output = os.path.join(args.output, "objects")
        shutil.copytree(object_input, object_output)

    if args.move_scenes:
        print("Moving scenes")
        scene_output_dir = os.path.join(args.output, "data")
        os.mkdir(scene_output_dir)

        for scene in scenes:
            print("Moving scene: {0}".format(scene))
            scene_input = os.path.join(args.scene_dir, scene)
            scene_output = os.path.join(scene_output_dir, scene)

            os.mkdir(scene_output)

            data_input = os.path.join(scene_input, "data")
            data_output = os.path.join(scene_output, "data")
            shutil.copytree(data_input, data_output)

            scene_meta_input = os.path.join(scene_input, "scene_meta.yaml")
            scene_meta_output = os.path.join(scene_output, "scene_meta.yaml")
            shutil.copyfile(scene_meta_input, scene_meta_output)


if __name__ == "__main__":
    main()