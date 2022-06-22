
# Using OptiTrack output to generate GT semantic segmentation

Basically, we take the optitrack which tracks markers, and we need to render the models in those poses, and then collect GT semantic segmentation data like that.

Current Objects:
 1. Cheerios Box (15.4oz)
 2. Campbell's Tomato Soup Can (10.75 oz)
 3. Campbell's Clam Chowder Can (10.5 oz)
 4. French's Yellow Mustard (14 oz)
 5. Poptarts Brown Sugar Cinammon Box (12 pc)
 6. Spam can (12 oz)
 7. StarKist Chunk Light Tuna Can (5 oz)
 8. Extra Toasty Cheez-It Box (12.4 oz)

Objects we hope to have soon:
 1. Expo markers of different color (Planning to use YCB Dataset)
 2. Scissors (6 inch and 8 inch, Scotch)

Final dataset output:
 * `objects` folder
 * `cameras` folder
 * `scenes` folder certain data:
 	 * `scenes/<scene number>/data/` folder
 	 * `scenes/<scene number>/scene_meta.yaml` metadata

# How to run
 1. Setup
	 1. Place ARUCO marker near origin (doesn't actually matter where it is anymore, but makes sense to be near opti origin)
	 2. Calibrate Opti (if you want, don't need to do this everytime, or else extrinsic changes)
	 3. Place a single marker in the center of the aruco marker, use this to compute the aruco -> opti transform
		 * Place the marker position into `calculate_extrinsic/aruco_marker.txt`
	 4. Place makrers on the corners of the aruco marker, use this to compute the aruco -> opti transform as well
	 	 * Place marker positions into `calculate_extrinsic/aruco_corners.yaml`
 2. Record Data
     1. ARUCO Calibration
	 2. Data collection
 3. Process Data
	 1. Clean raw opti poses
	 2. Sync opti poses with frames
	 3. Calculate extrinsic (virtual to real camera)
	 4. Manually annotate first frame object poses
	 5. Recover all frame object poses and verify correctness
	 6. Generate semantic labeling

