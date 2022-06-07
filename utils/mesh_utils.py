import numpy as np
import open3d as o3d

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

def convert_mesh_uvs_to_colors(mesh):
    triangle_indices = np.array(mesh.triangles)
    triangle_uvs = np.array(mesh.triangle_uvs)

    vertex_colors = np.array(mesh.vertex_colors)

    indices = triangle_indices.flatten()

    vertex_uvs_as_colors = np.zeros_like(vertex_colors)
    vertex_uvs_as_colors[:,:2][indices] = triangle_uvs

    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_uvs_as_colors)

    return mesh

def uniformly_sample_mesh_with_textures_as_colors(mesh, texture, number_of_points):
    pcld = mesh.sample_points_uniformly(number_of_points=number_of_points)
    
    uvs = np.array(pcld.colors)

    texture_y, texture_x, _ = texture.shape
    texture = texture.reshape((-1, 3))

    uvs_flattened = np.floor(uvs[:,0] * texture_x + np.floor(( 1. - uvs[:,1]) * texture_y) * texture_x).astype(np.int)

    colors = (texture[uvs_flattened] / 255.).astype(np.float32)
    pcld.colors = o3d.utility.Vector3dVector(colors)

    return pcld
    