"""
Cleans the exported tracking data output of the OptiTrack
"""

import cv2
import numpy as np
import shutil
import yaml

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from utils.constants import IPHONE_DEPTH_WIDTH, IPHONE_DEPTH_HEIGHT

class IPhoneDataProcessor():
    def __init__(self):
        pass

    @staticmethod
    def process_iphone_scene_data(scene_dir):

        iphone_data_input = os.path.join(scene_dir, "iphone_data")
        assert(os.path.isdir(iphone_data_input))

        frames = os.listdir(iphone_data_input)
        frame_ids = [f[:f.find(".")] for f in frames if ".jpeg" in f]
        frame_ids = sorted([int(f) for f in frame_ids])

        timestamp_data_file = os.path.join(scene_dir, "timestamps.txt")
        with open(timestamp_data_file, "r") as f:
            timestamp_data = f.readline()

        timestamp_data = timestamp_data.rstrip().split(",")

        frame_timestamps = [float(t) for t in timestamp_data[:-1]]
        capture_start_frame = int(timestamp_data[-1])

        assert(len(frame_timestamps) == len(frame_ids))

        #output files
        data_raw_output = os.path.join(scene_dir, "data_raw")

        if not os.path.isdir(data_raw_output):
            os.mkdir(data_raw_output)

        camera_data_output = open(os.path.join(scene_dir, "camera_data.csv"))
        camera_time_break_output = open(os.path.join(scene_dir, "camera_time_break.csv"))
        scene_metadata_output = os.path.join(scene_dir, "scene_meta.yaml")

        camera_data_output.write("Frame,Timestamp\n")
        camera_time_break_output.write("Calibration Start ID,Calibration End ID,Capture Start ID\n")

        for frame_id, frame_timestamp in zip(frame_ids, frame_timestamps):
            camera_data_output.write("{0},{1}\n".format(frame_id, frame_timestamp))

        camera_time_break_output.write("{0},{1},{2}\n".format(0, capture_start_frame, capture_start_frame))

        camera_data_output.close()
        camera_time_break_output.close()

        for frame_id in frame_ids:
            color_old = os.path.join(iphone_data_input, "{0}.jpeg".format(frame_id))
            color_new = os.path.join(data_raw_output, "{0}_color.jpg".format(str(frame_id).zfill(5)))
            shutil.copyfile(color_old, color_new)

            depth_old = os.path.join(iphone_data_input, "{0}.bin".format(frame_id))
            depth_new = os.path.join(data_raw_output, "{0}_depth.png".format(str(frame_id).zfill(5)))
            depth_arr = []

            with open(depth_old, 'rb') as f:
                byte = f.read(2)

                while byte != b"":
                    x = int.from_bytes(byte, "little", signed=False)
                    depth_arr.append(x)
                    byte = f.read(2)

            depth = np.array(depth_arr).astype(np.uint16)
            depth = depth.reshape((IPHONE_DEPTH_HEIGHT, IPHONE_DEPTH_WIDTH))

            cv2.imwrite(depth_new, depth)

        scene_metadata = {}
        scene_metadata["cam_scale"] = 0.001
        scene_metadata["camera"] = "iphone_camera1"
        scene_metadata["objects"] = []

        with open(scene_metadata_output, "w") as f:
            yaml.dump(scene_metadata, f)

        print("Transfered data into Azure Kinect format.")
        print("PLEASE fill the objects field in the scene_metadata.yaml")

        
        

