# Scenes

Each scene contains 2 folders:
1) data
2) objects

`data` is the true output for our dataset in terms of per-frame data. The other stuff is what we use to generate `data`.

`data` contains frames of format `00001_color.jpg`, `00001_depth.png`, `00001_label.png`, `00001_meta.txt`. 
- `00001_color.jpg`: color image (3 channel 8 bit) (comes from camera)
- `00001_depth.png`: depth image (16 bit 1 channel) (comes from camera, aligned to color)
- `00001_label.png`: ground truth semantic segmentation (1 channel 8 bit). We generate this using a virtual camera.
- `00001_meta.txt`: ground truth poses of objects in the scene. We calculate this using OptiTrack.
-- also contains `cam_scale`, units of depth image (multiply in order to retrieve meters)

`objects` contains folders named by the object id they correspond to.
Each object folder contains a `marker_positions.csv`:
- this defines the markers positions in the object coordinate system. We use these to recover object pose
-- this is defined per scene since the marker positions may change between scenes, since we have to remove them to record the camera data
-- the object coordinate system is defined in `objects/\<object name\>/\<object name\>.obj`

Each scene also contains a file:
`camera_poses.csv`:
- this contains the per-frame pose of the camera. The per-frame pose of each object in the camera frame can be easily solved for using this.

Each scene also contains a file:
`scene_meta.yaml`:
- this contains scene level metadata
-- camera intrinsics
-- which objects are in this scene
-- random stuff like time of day?