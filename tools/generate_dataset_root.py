"""
Generate actual dataset output.
"""

import argparse
import random
import numpy as np

import os, sys

from sklearn.model_selection import train_test_split 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

ROOT_TRAIN_DATA_LIST_FILE = "train_data_list.txt"
ROOT_TEST_DATA_LIST_FILE = "test_data_list.txt"
SKIP_FRAMES_NUM = 2
TRAIN_TEST_SPLIT_PERCENTAGE = 0.8
SCENE_NAME = "6d_test_dataset_17_scenes" #modify yourself

def main():
    random.seed(2022)
    np.random.seed(2022)
    parser = argparse.ArgumentParser(description='Generate data roots')
    args = parser.parse_args()

    #scenes_dir = os.path.join(dir_path, "..", SCENE_NAME)
    scenes_dir = os.path.join("F:", SCENE_NAME)
    dataset_root_dir = os.path.join(dir_path, "..", "toolbox", "root")
    folder_path_list = [os.path.join(scenes_dir, x) for x in os.listdir(scenes_dir)]
    random.shuffle(folder_path_list)
    train_folder_path_list = folder_path_list[:int(len(folder_path_list) * TRAIN_TEST_SPLIT_PERCENTAGE)]
    test_folder_path_list = folder_path_list[int(len(folder_path_list) * TRAIN_TEST_SPLIT_PERCENTAGE):]

    with open(os.path.join(dataset_root_dir, ROOT_TRAIN_DATA_LIST_FILE), 'w') as f:
        for each_scene_path in train_folder_path_list:
            data_dir = os.path.join(each_scene_path, "data")
            upper_limits = len(os.listdir(data_dir)) // 5
            ids = [str(x).zfill(5) for x in list(range(0, upper_limits, SKIP_FRAMES_NUM))]
            file_names = [(str(os.path.basename(os.path.normpath(each_scene_path))) + "/data/" + each_id +'\n') for each_id in ids]
            f.writelines(file_names)

    with open(os.path.join(dataset_root_dir, ROOT_TEST_DATA_LIST_FILE), "w") as f:
        for each_scene_path in test_folder_path_list:
            data_dir = os.path.join(each_scene_path, "data")
            upper_limits = len(os.listdir(data_dir)) // 5
            ids = [str(x).zfill(5) for x in list(range(0, upper_limits, SKIP_FRAMES_NUM))]
            file_names = [(str(os.path.basename(os.path.normpath(each_scene_path))) + "/data/" + each_id + "\n") for each_id in ids]
            f.writelines(file_names) 

if __name__ == "__main__":
    main()