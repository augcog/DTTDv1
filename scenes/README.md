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

Each scene also contains a file:
`scene_meta.yaml`:
- this contains scene level metadata
-- camera intrinsics
-- camera depth scale
-- which objects are in this scene
-- random stuff like time of day?