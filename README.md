
# Using OptiTrack output to generate GT semantic segmentation

Basically, we take the optitrack which tracks markers, and we need to render the models in those poses, and then collect GT semantic segmentation data like that.

Current Problems:

 1. Frame Synchronization
     1. Need to verify with 3d recon
     2. Need to verify whole pipeline with synchronized frames
 2. Object collection
     1. What objects? where get YCB
 3. Data collection
     1. Gotta collect data
     2. Richmond Field Station?

Final dataset output:
- `objects` folder
- `scenes` folder certain data:
-- `scenes/<scene number>/data/` folder
-- `scenes/<scene number>/scene_meta.yaml` metadata

# How to run

 1. Record Data
	 1. Shake Calibration
     2. ARUCO Calibration
	 3. Data collection
 2. Process Data
	 1. Clean raw opti poses
	 2. Sync opti poses with frames
	 3. calculate extrinsic (virtual to real camera)
	 4. Manually annotate first frame object poses
	 5. recover all frame object poses and verify correctness
	 6. Generate semantic labeling

