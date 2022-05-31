
# Using OptiTrack output to generate GT semantic segmentation

Basically, we take the optitrack which tracks markers, and we need to render the models in those poses, and then collect GT semantic segmentation data like that.

Current Problems:

- Need to figure out how to use the OptiTrack output to parameterize pose of objects in the scene
-- Probably going to have to use the marker positions and recover the pose ourselves. OptiTrack pose output uses incorrect center of object and arbitrary object coordinate system.
-- New plan: Don't use OptiTrack to get object pose estimation. Instead, manually annotate pose in first frame (by hand), then just use camera tracking to recover poses in all frames.
--- In which case, I need to write some sort of pose annotator tool (which is going to be fun :))
- Need to figure out how to use the OptiTrack output to parameterize pose of camera in the scene
-- Using the new calibration procedure that Allen proposed, we can use the OptiTrack pose. However, we still need to figure out how we're going to do the objects.
- Need to write the renderer to collect GT semantic labeling (should be pretty straight forward)
- Need to have 3D models for our objects
-- and Marker positions (for each scene) (NOTE: if we do manual annotation, we don't need these marker positions)

Final dataset output:
- `objects` folder
- `scenes` folder certain data:
-- `scenes/<scene number>/data/` folder
-- `scenes/<scene number>/scene_meta.yaml` metadata