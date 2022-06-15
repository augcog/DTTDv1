"""
Reconstructs output data from SaveFrame.cpp using Open3D's Integration
"""

#xyzw (quaternion), xyz (translation)
#rot_mat = scipy.spatial.transform.Rotation.from_quat(xyzw).as_matrix()
#affine = rot_mat || xyz (4x4, place a 0001 vector as the bottom row)
#affine is camera -> opti
#invert affine to get opti -> camera, this is what we will pass into integrate

import open3d as o3d 
import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
#from pyk4a import PyK4A, CalibrationType
from scipy.spatial.transform import Rotation as R
import cv2

def main():

    config = dict()
    config.update({"tsdf_cubic_size":0.5})
    config.update({"max_depth": 5.00})
    config.update({"voxel_size": 0.007})
    config.update({"max_depth_diff": 5.00})
    config.update({"python_multi_threading": True})
    

    data = pd.read_csv("new_sync_data.csv")

    rgb_images = os.path.join(os.getcwd(), "output", "color_image")
    depth_images = os.path.join(os.getcwd(), "output", "depth_image")
    transforms = rgb_images

    extrinsic = np.loadtxt("extrinsic.txt")

    #k4a = PyK4A()
    #k4a.start()
    #cam_intr = k4a.calibration.get_camera_matrix(CalibrationType.COLOR)
    cam_intr = [[614.81542969, 0., 638.14129639],
                [0., 614.67016602, 368.83706665], 
                [0., 0., 1.]]

    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(1280, 720, cam_intr[0][0], cam_intr[1][1], cam_intr[0][2], cam_intr[1][2])

    print("===================================")
    print("===================================")
    print("Reconstructing using this intrinsic matrix:")
    print(intr.intrinsic_matrix)
    print("===================================")
    print("===================================")
    for key in config.keys():
        print(key, ": ", config[key])
    print("===================================")
    print("===================================")

    volume = o3d.integration.ScalableTSDFVolume(
        voxel_length= config["voxel_size"],
        sdf_trunc=config["voxel_size"] * 5,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8)

    lst = list(data.index)[300:600]

    offset = 0

    for frame_id in lst:	
        if data.iloc[frame_id].isnull().any():
            continue
        #if op_sync_df.iloc[frame_id].
        az_frame_id = data.iloc[frame_id]["Frame"].astype(int)
        print("processing frame id: ", frame_id, "/{}, realframe is {}".format(len(lst), az_frame_id))

        #depth_test = cv2.imread(os.path.join(depth_images, "{}.png".format(az_frame_id)), cv2.IMREAD_UNCHANGED)

        color_raw = o3d.io.read_image(os.path.join(rgb_images, "{}.jpg".format(az_frame_id)))
        depth_raw = o3d.io.read_image(os.path.join(depth_images, "{}.png".format(az_frame_id)))

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, depth_trunc=config["max_depth"], convert_rgb_to_intensity=False)

        cam_pose = np.zeros((4,4))

        _,_,x,y,z,w,xt,yt,zt = data.iloc[frame_id + offset]
        rot_mat = R.from_quat([x,y,z,w]).as_matrix()

        cam_pose[:3, :3] = rot_mat
        cam_pose[:3, 3] = np.array([xt,yt,zt])
        cam_pose[3,3] = 1
        cam_pose = cam_pose @ np.linalg.inv(extrinsic)
        cam_pose = np.linalg.inv(cam_pose)

        volume.integrate(rgbd_image, intr, cam_pose)


    print("Finished Integrating")
    
    ''''
    point_cloud= volume.extract_point_cloud()
    o3d.io.write_point_cloud("pointcloud2.pcd",point_cloud)
    '''

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh("mesh.ply", mesh)

    print("Saved mesh to mesh.ply")



if __name__ == "__main__":
    main()