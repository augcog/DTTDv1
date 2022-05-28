import pandas as pd

class CameraSimulator():
    def __init__(self, camera_positions_csv, camera_poses_csv):
        positions = pd.read_csv(camera_positions_csv)
        poses = pd.read_csv(camera_poses_csv)