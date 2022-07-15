"""
Generates points.xyz for each object in objects.
"""

import argparse
import numpy as np

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from utils.object_utils import load_all_object_meshes

def write_points(f, points):
    out_file = open(f, "w")

    for xyz in points:
        out_file.write("{0} {1} {2}\n".format(xyz[0], xyz[1], xyz[2]))

    out_file.close()

def main():
    parser = argparse.ArgumentParser(description='Sample object pclds and put into points.xyz')
    parser.add_argument('--num_points', type=int, default=50000)
    args = parser.parse_args()

    objects_dir = os.path.join(dir_path, "..", "objects")

    object_meshes = load_all_object_meshes()

    for obj_id, obj_data in object_meshes.items():
        obj_name, obj_mesh = obj_data["name"], obj_data["mesh"]
        pcld = obj_mesh.sample_points_uniformly(number_of_points=args.num_points)
        pts = np.array(pcld.points)

        output_file = os.path.join(objects_dir, obj_name, "points.xyz")

        write_points(output_file, pts)

if __name__ == "__main__":
    main()