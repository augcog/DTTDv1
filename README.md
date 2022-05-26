
# Using OptiTrack output to generate GT semantic segmentation

Basically, we take the optitrack which tracks markers, and we need to render the models in those poses, and then collect GT semantic segmentation data like that.

Current Problems:

- Need to figure out how to use the OptiTrack output to parameterize pose of objects in the scene
- Need to figure out how to use the OptiTrack output to parameterize pose of camera in the scene
-- Probably going to have to use the marker positions and recover the pose ourselves. OptiTrack pose output uses incorrect center of object and arbitrary object coordinate system.
- Need to write the renderer to collect GT semantic labeling (should be pretty straight forward)
- Need to have 3D models for our objects
-- and Marker positions (for each scene)
- Need to have 3D positions of camera markers relative to sensor
-- since markers don't change, only need to do this once.