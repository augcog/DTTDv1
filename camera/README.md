# Camera

- `Extrinsic.txt` is the extrinsic transform from marker center of mass (the thing OptiTrack tracks) to the camera sensor
-- I have Jack working on writing the code to find this extrinsic using the ARUCO calibration.
- `positions.csv` is the postition of the markers in the camera coordinate system. I found these using a ruler. Probably not that accurate. Not necessary since we are using ARUCO calibration and OptiTrack pose.
