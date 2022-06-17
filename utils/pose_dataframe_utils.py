import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

def convert_pose_df_to_dict(pose_df):

    pose_out = {}

    for index, opti_pose_row in pose_df.iterrows():
        frame_id = opti_pose_row["Frame"].astype(int)

        opti_quat = np.array(opti_pose_row[["camera_Rotation_X","camera_Rotation_Y","camera_Rotation_Z","camera_Rotation_W"]])
        opti_translation = np.array(opti_pose_row[["camera_Position_X","camera_Position_Y","camera_Position_Z"]])

        opti_rot = R.from_quat(opti_quat).as_matrix()

        opti_pose = np.hstack((opti_rot, np.expand_dims(opti_translation, -1)))
        bot_row = np.array([[0, 0, 0, 1]])
        opti_pose = np.vstack((opti_pose, bot_row))

        pose_out[frame_id] = opti_pose

    return pose_out

def convert_pose_dict_to_df(pose_dict):
    frames = np.expand_dims(np.array(list(pose_dict.keys())).astype(int), -1)
    poses = np.array(list(pose_dict.values())).astype(np.float32)
    rots = poses[:,:3,:3]
    ts = poses[:,:3,3]
    quats = R.from_matrix(rots).as_quat()

    rows = np.hstack((frames, quats, ts))
    columns = ["Frame", "camera_Rotation_X","camera_Rotation_Y","camera_Rotation_Z","camera_Rotation_W", "camera_Position_X","camera_Position_Y","camera_Position_Z"]
  
    df = pd.DataFrame(rows, columns=columns)
    df["Frame"] = df["Frame"].astype(int)

    return df

