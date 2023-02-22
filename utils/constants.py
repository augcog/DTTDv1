
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

EXTRINSICS_DIR = os.path.join(dir_path, "..", "extrinsics_scenes")
SCENES_DIR = os.path.join(dir_path, "..", "scenes")

OPTI_FRAMERATE = 60


# Azure Kinect provides aligned color and depth
AZURE_KINECT_COLOR_HEIGHT = 720
AZURE_KINECT_COLOR_WIDTH = 1280

AZURE_KINECT_DEPTH_HEIGHT = 720
AZURE_KINECT_COLOR_WIDTH = 1280

# IPhone data will need to be aligned by us (off by a resolution factor)
IPHONE_COLOR_HEIGHT = 1440
IPHONE_COLOR_WIDTH = 1920

IPHONE_DEPTH_HEIGHT = 240
IPHONE_DEPTH_WIDTH = 320

ARKit_IPHONE_DEPTH_HEIGHT = 192
ARkit_IPHONE_DEPTH_WIDTH = 256