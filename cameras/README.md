# Cameras

Each camera contains:
 * `extrinsic.txt` is the extrinsic transform from marker center of mass (the thing OptiTrack tracks) to the camera sensor
     * I have Jack working on writing the code to find this extrinsic using the ARUCO calibration.

Some cameras contain:
 * `intrinsic.txt` is the camera intrinsic matrix. Used for ARUCO calibration.
 * `distortion.txt` is the distortion coefficients. Used for ARUCO calibration.

