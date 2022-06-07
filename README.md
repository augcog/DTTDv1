
# Using OptiTrack output to generate GT semantic segmentation

Basically, we take the optitrack which tracks markers, and we need to render the models in those poses, and then collect GT semantic segmentation data like that.

Current Problems:

 1. Frame Synchronization
     1. Need to verify with 3d recon
     2. Need to verify whole pipeline with synchronized frames
	 3. Looks like the frame synchronization looks okay!, still testing with a 3d recon
 2. Object collection
     1. What objects? where get YCB
	 2. Collected some objects
 3. Object 3D Reconstruction
	 1. How do we collect a high quality, full 3d reconstruction of all of our objects?
 4. Data collection
     1. Gotta collect data
     2. Richmond Field Station?

Final dataset output:
- `objects` folder
- `scenes` folder certain data:
-- `scenes/<scene number>/data/` folder
-- `scenes/<scene number>/scene_meta.yaml` metadata

# How to run
 1. Setup
	 1. Place ARUCO marker near origin (doesn't actually matter where it is anymore, but makes sense to be near opti origin)
	 2. Calibrate Opti (if you want, don't need to do this everytime, or else extrinsic changes)
	 3. Place a single marker in the center of the aruco marker, use this to compute the aruco -> opti transform
		 * Place the marker position into `camera/calculate_extrinsic/aruco_marker.txt`
 2. Record Data
	 1. Shake Calibration
     2. ARUCO Calibration
	 3. Data collection
 3. Process Data
	 1. Clean raw opti poses
	 2. Sync opti poses with frames
	 3. Calculate extrinsic (virtual to real camera)
	 4. Manually annotate first frame object poses
	 5. Recover all frame object poses and verify correctness
	 6. Generate semantic labeling

