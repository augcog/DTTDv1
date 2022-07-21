"""
Cleans the exported tracking data output of the OptiTrack
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from utils.constants import IPHONE_DEPTH_HEIGHT, IPHONE_DEPTH_WIDTH

class IPhoneDataProcessor():
    def __init__(self):
        pass

    @staticmethod
    def process_iphone_scene_data(scene_dir):

        depth_data = os.path.join(scene_dir, "depth.txt")
        assert(os.path.isfile(depth_data))

        data_raw_output = os.path.join(scene_dir, "data_raw")
        assert(os.path.isdir(data_raw_output))

        frames = os.listdir(data_raw_output)
        frame_ids = [f[:f.find("_")] for f in frames]
        frame_ids = sorted([int(f) for f in frame_ids])

        frame_idx = 0

        with open(depth_data, "r") as f:
            for depth_data_line in f:
                frame_id = frame_ids[frame_idx]

                depth_data = depth_data_line.rstrip().split(',')
                depth_data = [int(d) for d in depth_data]
                depth_data = np.array(depth_data).reshape((IPHONE_DEPTH_WIDTH, IPHONE_DEPTH_HEIGHT)).astype(np.uint16)
                depth_data = depth_data.T

                depth_data *= 30

                print(np.unique(depth_data[-2], return_counts=True))

                #cv2.imwrite("testtest.png", depth_data)
                
                frame_idx += 1

        assert(frame_idx == len(frame_ids))
