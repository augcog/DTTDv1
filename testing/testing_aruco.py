import argparse
import cv2
import numpy as np
import open3d as o3d
from pyk4a import PyK4A, CalibrationType
from scipy.spatial.transform import Rotation as R

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from utils.frame_utils import calculate_aruco_from_bgr_and_depth, load_bgr, load_depth, load_rgb
from utils.pointcloud_utils import pointcloud_from_rgb_depth, unproject_pixels

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test ARUCO.')
    parser.add_argument('--frame_dir', default='', type=str, help='A directory of frames to use (else uses plugged in camera)')

    args = parser.parse_args()

    # Modify camera configuration
    k4a = PyK4A()
    k4a.start()

    img_id = 0

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    # Load the predefined dictionary
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

    parameters =  cv2.aruco.DetectorParameters_create()

    camera_matrix = k4a.calibration.get_camera_matrix(CalibrationType.COLOR)
    camera_dist = k4a.calibration.get_distortion_coefficients(CalibrationType.COLOR)

    output_dir = os.path.join(dir_path, "aruco_vis")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    while True:
        if not args.frame_dir:
            # Get capture
            capture = k4a.get_capture()

            # # Get the color image from the capture
            color_image = capture.color[:,:,:3]
            depth_image = capture.transformed_depth

            color_image = np.ascontiguousarray(color_image)
            rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        else:
            try:
                color_image = load_bgr(args.frame_dir, img_id)
                rgb = load_rgb(args.frame_dir, img_id)
                depth_image = load_depth(args.frame_dir, img_id)
            except:
                print("out of frames at id {0}".format(img_id))
                break

        aruco_pose = calculate_aruco_from_bgr_and_depth(color_image, depth_image, 0.001, camera_matrix, camera_dist, dictionary, parameters)

        if aruco_pose:

            rvec, tvec, corners = aruco_pose

            camera_pcld = pointcloud_from_rgb_depth(rgb, depth_image, 0.001, camera_matrix, camera_dist)
            aruco_pcld = o3d.geometry.PointCloud()
            aruco_pcld.points = o3d.utility.Vector3dVector(tvec)
            o3d.io.write_point_cloud(os.path.join(output_dir, "{0}_camera.ply".format(img_id)), camera_pcld)
            o3d.io.write_point_cloud(os.path.join(output_dir, "{0}_aruco_origin.ply".format(img_id)), aruco_pcld)

            (rvec - tvec).any()  # get rid of that nasty numpy value array error
            cv2.aruco.drawDetectedMarkers(color_image, corners)  # Draw A square around the markers
            cv2.aruco.drawAxis(color_image, camera_matrix, camera_dist, rvec, tvec / 9, 0.01)  # Draw Axis

            # Display the resulting frame
            cv2.imshow("image", color_image)
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)  
            gray = cv2.GaussianBlur (gray, (5, 5), 5)

        img_id += 1

        # Press q key to stop
        if cv2.waitKey(1) == ord('q'):
            break
    
