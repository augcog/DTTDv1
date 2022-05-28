# Scenes

Each scene contains 2 folders:
1) data
2) objects

`data` contains frames of format `00001_color.jpg`, `00001_depth.png`, `00001_label.png`, `00001_meta.txt`. 
- `00001_color.jpg`: color image (3 channel 8 bit)
- `00001_depth.png`: depth image (16 bit 1 channel)
- `00001_label.png`: ground truth semantic segmentation (1 channel 8 bit). We generate this using a virtual camera.
- `00001_meta.txt`: ground truth poses of objects in the scene
-- also contains `cam_scale`, units of depth image (multiply in order to retrieve meters)

`objects` contains a `positions.csv`:
- this defines the markers positions in the object coordinate system. We use these to recover object pose
-- this is defined per scene since the marker positions may change between scenes, since we have to remove them to record the camera data

Each scene also contains a file:
`frame_marker_positions.csv`:
- this contains the per-frame marker positions for every marker in the scene. This is used in tandem with each object's `positions.csv` in order to get object poses.