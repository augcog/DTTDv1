"""
Cleans the exported tracking data output of the OptiTrack
"""

import csv
import numpy as np
import os
import pandas as pd
import yaml

class CameraPoseCleaner():
    def __init__(self):
        pass

    @staticmethod
    def load_from_file(cleaned_poses_path):
        df = pd.read_csv(cleaned_poses_path)
        return df

    @staticmethod
    def clean_camera_pose_file(scene_dir, write_cleaned_to_file=False):

        camera_pose_path = os.path.join(scene_dir, "camera_poses", "camera_poses.csv")
        camera_pose_cleaned_path = os.path.join(scene_dir, "camera_poses", "camera_poses_cleaned.csv")

        scene_metadata_file = os.path.join(scene_dir, "scene_meta.yaml")

        with open(scene_metadata_file, 'r') as file:
            scene_metadata = yaml.safe_load(file)

        camera_name = scene_metadata["camera"]

        header_rows = []
        rows = []

        with open(camera_pose_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row_id, row in enumerate(reader):

                if row_id == 3 or row_id == 5 or row_id == 6:
                    header_rows.append(row)

                elif row_id > 6:
                    rows.append(row)
                else:
                    continue

        headers = ['_'.join([x[i] for x in header_rows if len(x[i]) > 0]) for i in range(len(header_rows[0]))]
        headers = [h.replace(" ", "_").replace("(", "").replace(")", "") for h in headers]

        first_marker_column = min([i for (i, h) in enumerate(headers) if "Marker" in h])

        headers = headers[:first_marker_column]
        rows = [row[:first_marker_column] for row in rows]

        headers = [h.replace(camera_name, "camera") for h in headers]

        df = pd.DataFrame(rows, columns=headers)
        df.replace('', np.nan, inplace=True)
        df = df.dropna()

        if write_cleaned_to_file:
            df.to_csv(camera_pose_cleaned_path)

        return df
