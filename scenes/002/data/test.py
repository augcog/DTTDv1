import cv2
import os

# color_image_dir = "color_image"
# for f in os.listdir(color_image_dir):
#     abs_f = os.path.join(color_image_dir, f)

#     frame_id = f[:f.find(".")]
#     frame_ext = f[f.find("."):]

#     frame_out = frame_id.zfill(5) + "_color" + frame_ext
    
#     frame = cv2.imread(abs_f, cv2.IMREAD_UNCHANGED)

#     cv2.imwrite(frame_out, frame)

# depth_image_dir = "depth_image"
# for f in os.listdir(depth_image_dir):
#     abs_f = os.path.join(depth_image_dir, f)

#     frame_id = f[:f.find(".")]
#     frame_ext = f[f.find("."):]

#     frame_out = frame_id.zfill(5) + "_depth" + frame_ext
    
#     frame = cv2.imread(abs_f, cv2.IMREAD_UNCHANGED)

#     cv2.imwrite(frame_out, frame)

import cv2
import numpy as np

#frame = cv2.imread("00304_color.jpg")
frame = cv2.imread("304.jpg")
cv2.imshow("test", frame)
cv2.waitKey(0)

#camera_intrinsic_matrix = np.loadtxt(r'C:\Users\OpenARK\Desktop\adam\6d-pose-dataset\camera\intrinsic.txt')
#camera_distortion_coefficients = np.loadtxt(r'C:\Users\OpenARK\Desktop\adam\6d-pose-dataset\camera\distortion.txt')

camera_intrinsic_matrix = np.array([[614.81542969,   0.,         638.14129639],
 [  0.,         614.67016602, 368.83706665],
 [  0.,           0.,           1.,        ]]) 
 
camera_distortion_coefficients = np.array([4.16982830e-01, -2.29386592e+00,  9.83755919e-04, -4.50664316e-04,
  1.23773777e+00,  2.97605455e-01, -2.12851882e+00,  1.17534304e+00])

print(camera_intrinsic_matrix.shape, camera_intrinsic_matrix.dtype)
print(camera_distortion_coefficients.shape, camera_distortion_coefficients.dtype)

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()
corners, ids, rejected_img_points = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters, cameraMatrix=camera_intrinsic_matrix, distCoeff=camera_distortion_coefficients)

print(corners)
print(ids)