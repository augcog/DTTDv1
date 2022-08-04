import argparse

import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from data_processing import CameraPoseCleaner, CameraPoseSynchronizer
from pose_refinement import OptiARIMASmoother, OptiKFSmoother, OptiSavgolSmoother
from utils.constants import EXTRINSICS_DIR, SCENES_DIR
from utils.datetime_utils import get_latest_str_from_str_time_list

def main():
    parser = argparse.ArgumentParser(description='Synchronize optitrack poses with frames')
    parser.add_argument('--scene_name', type=str, help='name of scene')
    parser.add_argument('--extrinsic', default=False, action="store_true", help='processing a extrinsic scene')
    args = parser.parse_args()

    if not args.scene_name and not args.extrinsic:
        print("Must be a scene capture (indicate a scene_name) or an extrinsic capture (use --extrinsic flag).")
        exit(-1)

    if args.extrinsic and args.scene_name:
        scene_dir = os.path.join(EXTRINSICS_DIR, args.scene_name)
    elif args.extrinsic:
        extrinsic_scenes = list(os.listdir(EXTRINSICS_DIR))
        latest_extrinsic_scene = get_latest_str_from_str_time_list(extrinsic_scenes)

        print("using extrinsic scene {0}".format(latest_extrinsic_scene))
        
        scene_dir = os.path.join(EXTRINSICS_DIR, latest_extrinsic_scene)
    else:
        scene_dir = os.path.join(SCENES_DIR, args.scene_name)

    cam_pose_cleaner = CameraPoseCleaner()
    cleaned_poses = cam_pose_cleaner.clean_camera_pose_file(scene_dir, write_cleaned_to_file=False)

    cam_pose_smoother = OptiKFSmoother()
    smoothed_poses = cam_pose_smoother.smooth_opti_poses_kf(scene_dir, cleaned_poses, write_smoothed_to_file=False)

    # cam_pose_smoother_arima = OptiARIMASmoother()
    # smoothed_poses_arima = cam_pose_smoother_arima.smooth_opti_poses_arima(scene_dir, cleaned_poses, write_smoothed_to_file=False)

    cam_pose_smoother_savgol = OptiSavgolSmoother()
    smoothed_poses_savgol = cam_pose_smoother_savgol.smooth_opti_poses_savgol(scene_dir, cleaned_poses, write_smoothed_to_file=False)

    cam_pose_synchronizer = CameraPoseSynchronizer()
    smoothed_synchronized_df, total_offset, frame_ids = cam_pose_synchronizer.synchronize_camera_poses_and_frames(scene_dir, smoothed_poses, show_sync_plot=False, write_to_file=False, rewrite_images = False)

    notsmoothed_synchronized_df = cam_pose_synchronizer.get_synchronized_camera_poses_and_frames_with_known_offset(scene_dir, cleaned_poses, total_offset, frame_ids)

    # smoothed_arima_synchronized_df = cam_pose_synchronizer.get_synchronized_camera_poses_and_frames_with_known_offset(scene_dir, smoothed_poses_arima, total_offset, frame_ids)

    smoothed_savgol_synchronized_df = cam_pose_synchronizer.get_synchronized_camera_poses_and_frames_with_known_offset(scene_dir, smoothed_poses_savgol, total_offset, frame_ids)

    output_dir = os.path.join(scene_dir, "camera_poses_test")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    smoothed_synchronized_df.to_csv(os.path.join(output_dir, "camera_poses_smoothed.csv"))
    notsmoothed_synchronized_df.to_csv(os.path.join(output_dir, "camera_poses_orig.csv"))
    # smoothed_arima_synchronized_df.to_csv(os.path.join(output_dir, "camera_poses_smoothed_arima.csv"))
    smoothed_savgol_synchronized_df.to_csv(os.path.join(output_dir, "camera_poses_smoothed_savgol.csv"))

if __name__ == "__main__":
    main()
