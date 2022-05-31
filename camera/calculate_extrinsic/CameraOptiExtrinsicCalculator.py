import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation as R

from camera_pose_processing.CameraPoseSynchronizer import CameraPoseSynchronizer

class CameraOptiExtrinsicCalculator():
    def __init__(self, camera_intrinsic_matrix, camera_distortion_coefficients):
        self.camera_intrinsic_matrix = camera_intrinsic_matrix
        self.camera_distortion_coefficients = camera_distortion_coefficients

    @staticmethod
    def invert_affine(affine):
        assert(affine.shape == (4, 4))

        invert_affine = np.zeros((4, 4))
        invert_affine[:3,:3] = affine[:3,:3].T
        invert_affine[:3,3] = -affine[:3,3]
        invert_affine[3,3] = 1.

        return invert_affine

    #quats of shape (4xN)
    #return average quaternion
    @staticmethod
    def average_quaternion(quats):
        assert(quats.shape[0] == 4)

        q = quats @ quats.T

        w, v = np.linalg.eig(q)

        max_eigenvalue_idx = np.argmax(w)
        return v[:,max_eigenvalue_idx]

    @staticmethod
    def affine_matrix_from_rotvec_trans(rot_vec, trans):
        rot = R.from_rotvec(rot_vec)
        rmatrix = rot.as_matrix()
        trans = np.expand_dims(trans, -1)
        aff = np.hstack((rmatrix, trans))
        lrow = np.array([0,0,0,1])
        aff = np.vstack((aff, lrow))
        return aff

    #assumed that the optitrack origin is 4cm above the aruco marker taped to the ground
    @staticmethod
    def add_optitrack_aruco_offset(aff, vertical_offset=0.04):
        rot = aff[:3, :3]
        trans_offset = rot @ np.array([[0], [0], [vertical_offset]])
        aff[:3,3] += trans_offset
        return aff

    @staticmethod
    def calculate_camera_to_opti_transform(rot_vec, trans):
        aff = CameraOptiExtrinsicCalculator.affine_matrix_from_rotvec_trans(rot_vec, trans) #aruco -> camera
        aff = CameraOptiExtrinsicCalculator.add_optitrack_aruco_offset(aff) #aruco -> camera
        aruco_to_opti = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) #aruco -> opti (rot (3x3))

        opti_to_aruco = np.eye(4)
        opti_to_aruco[:3,:3] = np.linalg.inv(aruco_to_opti) #opti -> aruco (affine (4x4))

        affine_transform_opti = np.matmul(aff, opti_to_aruco) #1. opti -> aruco 2. aruco -> camera = opti -> camera
        return CameraOptiExtrinsicCalculator.invert_affine(affine_transform_opti) #camera -> opti

    def calculate_extrinsic(self, frames_dir, opti_poses_df, pose_synchronization):

        virtual_camera_to_opti_transforms = []
        camera_sensor_to_opti_transforms = []

        for frame_id, opti_pose_row_number in pose_synchronization.items():
            frame = cv2.imread(os.path.join(frames_dir, frame_id + ".jpg"))

            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
            parameters = cv2.aruco.DetectorParameters_create()
            corners, ids, rejected_img_points = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters, cameraMatrix=self.camera_intrinsic_matrix, distCoeff=self.camera_distortion_coefficients)
            
            if np.all(ids is not None):  # If there are markers found by detector

                #only should have 1 marker placed near origin of opti
                assert(len(ids) == 1)
                corners = corners[0]  # Iterate in markers
                # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 0.02, self.camera_intrinsic_matrix, self.camera_distortion_coefficients)

                camera_sensor_to_opti_transform = CameraOptiExtrinsicCalculator.calculate_camera_to_opti_transform(rvec, tvec)

                camera_sensor_to_opti_transforms.append(camera_sensor_to_opti_transform)

                #get optitrack pose for virtual camera
                opti_pose_row = opti_poses_df[opti_pose_row_number]

                opti_quat = np.array(opti_pose_row["camera_Rotation_X","camera_Rotation_Y","camera_Rotation_Z","camera_Rotation_W"])
                opti_translation = np.array(opti_pose_row["camera_Position_X","camera_Position_Y","camera_Position_Z"])

                opti_rot = R.from_quat(opti_quat).as_matrix()

                opti_pose = np.hstack((opti_rot, np.expand_dims(opti_translation, -1)))
                opti_pose = np.vstack((opti_pose, np.array([[0, 0, 0, 1]]).T))

                virtual_camera_to_opti_transforms.append(opti_pose)
            
            else:
                print("frame {0} did not successfully find ARUCO marker. Therefore it has been skipped.".format(frame_id))
            

        #extrinsic from virtual to camera sensor
        extrinsics = []

        for camera_to_opti, virtual_to_opti in zip(camera_sensor_to_opti_transforms, virtual_camera_to_opti_transforms):
            extrinsics.append([CameraOptiExtrinsicCalculator.invert_affine(camera_to_opti) @ virtual_to_opti])

        extrinsics = np.array(extrinsics)

        #make sure the extrinsics are relatively similar
        rotation_diffs = []
        translation_diffs = []
        for x in range(len(extrinsics) - 1):
            rotation_diffs.append(np.linalg.norm(extrinsics[x + 1][:3,:3] - extrinsics[x][:3,:3]))
            translation_diffs.append(np.linalg.norm(extrinsics[x + 1][:3,3] - extrinsics[x][:3,3]))

        print("diffs should be small (ideally 0)")
        print("rotation diffs (min, max, mean)", np.min(rotation_diffs), np.max(rotation_diffs), np.mean(rotation_diffs))
        print("translation diffs (min, max, mean)", np.min(translation_diffs), np.max(translation_diffs), np.mean(translation_diffs))

        #return average extrinsic
        translation = np.mean(extrinsics[:,:3,3])

        extrinsic_rots = extrinsics[:,:3,:3]
        extrinsic_quats = R.from_matrix(extrinsic_rots).as_quat()

        quat = CameraOptiExtrinsicCalculator.average_quaternion(extrinsic_quats)

        extrinsic = np.eye(4)
        extrinsic[:3,:3] = R.from_quat(quat).as_matrix()
        extrinsic[:3,3] = translation

        return extrinsic
