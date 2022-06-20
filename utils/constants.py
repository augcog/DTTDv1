
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

EXTRINSICS_DIR = os.path.join(dir_path, "..", "extrinsics_scenes")
SCENES_DIR = os.path.join(dir_path, "..", "scenes")

OPTI_FRAMERATE = 60