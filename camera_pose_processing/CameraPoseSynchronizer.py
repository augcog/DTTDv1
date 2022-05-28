import os
import pandas as pd


"""
Given a csv of camera poses from OptiTrack and a sequence of RGB images, perform synchronization
matching an RGB image with an OptiTrack pose. Uses the shake synchronization (TM) method.
"""
class CameraPoseSynchronizer():
    def __init__(self):
        pass

    """
    frames_dir is directory with frames inside
    poses is dataframe outputted by CameraPoseCleaner
    synchronization_frame_end: stop calculating and looking for peaks past this frame number.
    Basically, calculate synchronization only using frames before this frame number. 

    Returns dict: {key = frame id, value = pose row number}
    """
    @staticmethod
    def synchronize_camera_poses_and_frames(frames_dir, poses, synchronization_frame_end):

        #calculate the per-frame velocity using poses

        #calculate the per-frame pixel change using frames

        #match peaks

        #compute synchronization

        #match frames

        #return matching poses and camera frames

        return {}

    """
    If ya want to pass in a poses file
    """
    @staticmethod
    def synchronize_camera_poses_from_csv_and_frames(frames_dir, poses_csv, synchronization_frame_end):
        poses = pd.read_csv(poses_csv)
        return CameraPoseSynchronizer.synchronize_camera_poses_and_frames(frames_dir, poses, synchronization_frame_end)

