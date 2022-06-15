import sys
import cv2
import os
import numpy as np
import pandas as pd
import datetime
import keyboard
from scipy.spatial.transform import Rotation as R

from sklearn.metrics import f1_score
sys.path.insert(1, '../')
from pyk4a import PyK4A, CalibrationType


from tomlkit import date

sys.path.insert(1, '../')

CALI_KEY_START = 'c'
CALI_KEY_STOP = 'C'
CAP_KEY_START = 's'
CAP_KEY_STOP = 'S'
QUIT_KEY = 'q'
color_image_directory = os.path.join(os.getcwd(), "output",  "color_image")
depth_image_directory = os.path.join(os.getcwd(), "output",  "depth_image")

os.makedirs(color_image_directory, exist_ok=True)
os.makedirs(depth_image_directory, exist_ok=True)

aruco_to_opti = np.loadtxt("ARUCO_TO_OPTI.txt")
virtual_to_camera = np.loadtxt("extrinsic_new.txt")

def collect_data(f, img_id, color_image_directory, depth_image_directory):
    capture = k4a.get_capture()
    cur_timestamp = str(datetime.datetime.now())

    color_image = capture.color[:,:,:3]
    depth_image = capture.transformed_depth

    color_image = np.ascontiguousarray(color_image)
    corners, ids, rejectedCandidates = cv2.aruco.detectMarkers(color_image, dictionary, parameters=parameters)
    xyz_pos = np.array([0, 0, 0])
    if np.all(ids is not None):  # If there are markers found by detector
        for i in range(0, len(ids)):  # Iterate in markers
            # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, camera_matrix,
                                                                        camera_dist)
            rvec = np.squeeze(rvec, axis = 1)
            tvec = np.squeeze(tvec, axis = 1)
            markerPoints = np.squeeze(markerPoints, axis=1)
            rotmat = R.from_rotvec(rvec).as_matrix() # rvec -> rotation matrix
            aruco_to_sensor = np.zeros((4,4)) # aruco -> sensor
            aruco_to_sensor[:3, :3] = rotmat
            aruco_to_sensor[:3, 3] = tvec * 9
            aruco_to_sensor[3,3] = 1
            sensor_to_aruco = np.linalg.inv(aruco_to_sensor) # sensor -> aruco 
            sensor_to_opti = aruco_to_opti @ sensor_to_aruco # sensor -> opti
            #virtual_to_opti = sensor_to_opti @ virtual_to_camera
            # xyz_pos = virtual_to_opti[:3,-1]
            xyz_pos = sensor_to_opti[:3,-1] # sensor's pos in opti space
            #cv2.aruco.drawDetectedMarkers(color_image, corners)  # Draw A square around the markers
            #cv2.aruco.drawAxis(color_image, camera_matrix, camera_dist, rvec, tvec, 0.01)  # Draw Axis
    
    threshold = 1.0 - np.count_nonzero(depth_image)/(depth_image.shape[0] * depth_image.shape[1])
    # if threshold > 0.7:
    #     continue

    cv2.imshow("Color Image", color_image)
    cv2.imshow("Depth Image", depth_image)

    color_image_filename = "{}.jpg".format(img_id)
    depth_image_filename = "{}.png".format(img_id)

    cv2.imwrite(os.path.join(color_image_directory, color_image_filename), color_image)
    cv2.imwrite(os.path.join(depth_image_directory, depth_image_filename), depth_image)
    x, y, z = xyz_pos
    f.write("{},{}, {}, {}, {}, {}\n".format(img_id, cur_timestamp, threshold, x, y, z))

if __name__ == "__main__":

    ### configuration ###
    # Initialize the library, if the library is not found, add the library path as argument
    k4a = PyK4A()
    k4a.start()
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

    parameters =  cv2.aruco.DetectorParameters_create()

    camera_matrix = k4a.calibration.get_camera_matrix(CalibrationType.COLOR)
    camera_dist = k4a.calibration.get_distortion_coefficients(CalibrationType.COLOR)
    # print(device_config)
    print(camera_matrix, camera_dist)

    # Start device
    video_filename = "output.mkv"
    img_id = 0

    cv2.namedWindow("Color Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth Image", cv2.WINDOW_NORMAL)

    print("--------------------------------------")
    print("Begin generating data at {}".format(str(datetime.datetime.now())))
    print("--------------------------------------")

    calib_btn = False
    scan_btn = False

    calibration_start_img_id = -1
    calibration_stop_img_id = -2
    capture_start_img_id = -3
    capture_stop_img_id = -4

    data_file = open("az_data.csv", "w")

    while True:

        # keyboard.on_press_key(CALI_KEY, lambda _:on_button())

        if cv2.waitKey(1) == ord(CALI_KEY_START) and not calib_btn: 
            calib_btn = True
            print("Calibration start img_id is {}".format(img_id))
            calibration_start_img_id = img_id
        elif cv2.waitKey(1) == ord(CALI_KEY_STOP) and calib_btn: 
            calib_btn = False
            print("Calibration stop img_id is {}".format(img_id))
            calibration_stop_img_id = img_id

        if cv2.waitKey(1) == ord(CAP_KEY_START) and not scan_btn: 
            scan_btn = True
            print("Capturing start img_id is {}".format(img_id))
            capture_start_img_id = img_id
        elif cv2.waitKey(1) == ord(CAP_KEY_STOP) and scan_btn: 
            scan_btn = False
            print("Capturing stop img_id is {}".format(img_id))
            capture_stop_img_id = img_id
        


        collect_data(data_file, img_id, color_image_directory, depth_image_directory)
        img_id += 1

        with open("az_time_break.csv", "w") as f:
            f.write("{}\n{}\n{}\n{}".format(calibration_start_img_id, calibration_stop_img_id, capture_start_img_id, capture_stop_img_id))
            
        #Press q key to stop
        if cv2.waitKey(1) == ord(QUIT_KEY): 
            break

    data_file.close()
        

    

