"""
Given objects and their poses in camera frame, generate semantic segmentation.
"""

import open3d as o3d

class SemanticLabelingGenerator():
    def __init__(self, objects, annotated_object_poses, camera_intrinsic_matrix, camera_distortion_coefficients):
        self._objects = {}
        for obj_id, obj_data in objects.items():
            self._objects[obj_id] = {"name" : obj_data["name"]}
            self._objects[obj_id]["mesh"] = o3d.geometry.TriangleMesh(obj_data["mesh"]) #copy the geometry
        self.camera_intrinsic_matrix = camera_intrinsic_matrix
        self.camera_distortion_coefficients = camera_distortion_coefficients

    """
    object_poses: {
        frame_id: {
            obj_id: pose
        }
    }
    """
    def generate_semantic_labels(self, object_poses):
