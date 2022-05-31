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


    @staticmethod
    def affinematrix(rot_vec_x, rot_vec_y, rot_vec_z, x, y, z):
        rot_vec = np.array([rot_vec_x, rot_vec_y, rot_vec_z])
        rot = R.from_rotvec(rot_vec)
        rmatrix = rot.as_matrix()
        trans = np.array([[x], [y], [z]])
        trans_offset = rmatrix @ np.array([[0], [0], [0.04]])

        print(trans_offset)
        
        aff = np.hstack((rmatrix, trans + trans_offset))
        lrow = np.array([0,0,0,1])
        aff = np.vstack((aff, lrow))
        return aff

    #this function has several assumptions
    #relation of aruco marker coordinate system to optitrack coordinate system
    #vertical offset between the two (around 1-3cm)
    @staticmethod
    def calculate_camera_to_opti_transform(rot_vec_x, rot_vec_y, rot_vec_z, x, y, z):
        
        #TODO: currently, Jack is writing this function
        aff = CameraOptiExtrinsicCalculator.affinematrix(rot_vec_x, rot_vec_y, rot_vec_z, x, y, z) #aruco -> camera
        aruco_to_opti = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) #aruco -> opti
        opti_to_aruco = np.linalg.inv(aruco_to_opti) #opti -> aruco (3x3)
        #opti_to_aruco = aruco_to_opti
        opti_to_aruco = np.hstack((opti_to_aruco, np.array([[0], [0], [0]])))
        opti_to_aruco = np.vstack((opti_to_aruco, np.array([0,0,0,1]))) #opti -> aruco (affine)
        #print(opti_to_aruco.shape)
        #print(aff.shape)
        affine_transform_opti = np.matmul(aff, opti_to_aruco) #1. opti -> aruco 2. aruco -> camera = opti -> camera
        return CameraOptiExtrinsicCalculator.invert_affine(affine_transform_opti) #camera -> opti

    

    @staticmethod
    def calculate_extrinsic(frames_dir, opti_poses_df, pose_synchronization):
        #opti_poses: virtual camera -> opti
        opti_poses = []

        #aruco poses: aruco -> camera sensor
        aruco_poses = []
        for frame_id, opti_pose_row_number in pose_synchronization.items():
            frame = cv2.imread(os.path.join(frames_dir, frame + ".jpg"))

            #TODO: calculate aruco pose thing using rvec and tvec
            projection_matrix = np.array([[614.81542969, 0, 638.14129639],
            [0, 614.67016602, 368.83706665], [0, 0, 1]])
            distortion_coefficents = np.array([4.16982830e-01, -2.29386592e+00, 9.83755919e-04, -4.50664316e-04, 
            1.23773777e+00, 2.97605455e-01, -2.12851882e+00, 1.17534304e+00])

            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
            parameters = cv2.aruco.DetectorParameters_create()
            corners, ids, rejected_img_points = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters, cameraMatrix=projection_matrix, distCoeff=distortion_coefficents)
            
            if np.all(ids is not None):  # If there are markers found by detector
                for i in range(0, len(ids)):  # Iterate in markers
                # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                    rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, projection_matrix, distortion_coefficents)
                    (rvec - tvec).any()  # get rid of that nasty numpy value array error

                    


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
