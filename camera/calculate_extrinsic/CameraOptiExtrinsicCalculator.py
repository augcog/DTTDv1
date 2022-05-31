import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation as R

class CameraOptiExtrinsicCalculator():
    def __init__(self):
        pass

    @staticmethod
    def invert_affine(affine):
        assert(affine.shape == (4, 4))

        invert_affine = np.zeros((4, 4))
        invert_affine[:3,:3] = affine[:3,:3].T
        invert_affine[:3,3] = -affine[:3,3]
        invert_affine[3,3] = 1.

        return invert_affine

    #this function has several assumptions
    #relation of aruco marker coordinate system to optitrack coordinate system
    #vertical offset between the two (around 1-3cm)
    @staticmethod
    def calculate_camera_to_opti_transform(aruco_poses):

        #TODO: currently, Jack is writing this function
        
        return

    @staticmethod
    def calculate_extrinsic(frames_dir, opti_poses_df, pose_synchronization):
        #opti_poses: virtual camera -> opti
        opti_poses = []

        #aruco poses: aruco -> camera sensor
        aruco_poses = []
        for frame_id, opti_pose_row_number in pose_synchronization.items():
            frame = cv2.imread(os.path.join(frames_dir, frame + ".jpg"))

            #TODO: calculate aruco pose thing using rvec and tvec

            aruco_poses.append(aruco_pose)

            #get optitrack pose for virtual camera
            opti_pose_row = opti_poses_df[opti_pose_row_number]

            opti_quat = np.array(opti_pose_row["camera_Rotation_X","camera_Rotation_Y","camera_Rotation_Z","camera_Rotation_W"])
            opti_translation = np.array(opti_pose_row["camera_Position_X","camera_Position_Y","camera_Position_Z"])

            opti_rot = R.from_quat(opti_quat).as_matrix()

            opti_pose = np.hstack((opti_rot, np.expand_dims(opti_translation, -1)))
            opti_pose = np.vstack((opti_pose, np.array([[0, 0, 0, 1]]).T))

            opti_poses.append(opti_pose)
            

        #(Nx4x4)
        camera_to_opti_transforms = CameraOptiExtrinsicCalculator.calculate_camera_to_opti_transform(aruco_poses)

        #camera_to_opti_transforms: camera sensor -> opti

        extrinsics = []

        for camera_to_opti, virtual_to_opti in zip(camera_to_opti_transforms, opti_poses):
            extrinsics.append([CameraOptiExtrinsicCalculator.invert_affine(camera_to_opti) @ virtual_to_opti])

        extrinsics = np.array(extrinsics)

        #make sure the extrinsics are relatively similar
        rotation_diffs = []
        translation_diffs = []
        for x in range(len(extrinsics) - 1):
            rotation_diffs.append(np.linalg.norm(extrinsics[x + 1][:3,:3] - extrinsics[x][:3,:3]))
            translation_diffs.append(np.linalg.norm(extrinsics[x + 1][:3,3] - extrinsics[x][:3,3]))

        print("diffs should be small")
        print("rotation diffs (min, max, mean)", np.min(rotation_diffs), np.max(rotation_diffs), np.mean(rotation_diffs))
        print("translation diffs (min, max, mean)", np.min(translation_diffs), np.max(translation_diffs), np.mean(translation_diffs))

        #TODO: want to return the average extrinsic
        translation = np.mean(extrinsics[:,3])

        return np.eye(4)
