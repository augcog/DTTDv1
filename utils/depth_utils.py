import numpy as np
import pandas as pd

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

import utils.depth_map_utils as depth_map_utils

def fill_missing(
            dpt, cam_scale, scale_2_80m, fill_type='multiscale',
            extrapolate=False, show_process=False, blur_type='bilateral'
    ):
        dpt = dpt / cam_scale * scale_2_80m
        projected_depth = dpt.copy()
        if fill_type == 'fast':
            final_dpt = depth_map_utils.fill_in_fast(
                projected_depth, extrapolate=extrapolate, blur_type=blur_type,
                # max_depth=2.0
            )
        elif fill_type == 'multiscale':
            final_dpt, process_dict = depth_map_utils.fill_in_multiscale(
                projected_depth, extrapolate=extrapolate, blur_type=blur_type,
                show_process=show_process,
                max_depth=3.0
            )
        else:
            raise ValueError('Invalid fill_type {}'.format(fill_type))
        dpt = final_dpt / scale_2_80m * cam_scale
        return dpt

def norm2bgr(norm):
    norm = ((norm + 1.0) * 127).astype("uint8")
    return norm

def filter_depths_valid_percentage(depth_valid):

    rolling_window_size = 10

    rolling_max = pd.Series(depth_valid).rolling(rolling_window_size).max()
    rolling_max = rolling_max[rolling_window_size - 1:]

    rolling_max = np.pad(rolling_max, (0, len(depth_valid) - len(rolling_max)), 'edge')
    
    valid_threshold = 0.8 * rolling_max
    definitely_bad = depth_valid < valid_threshold

    print("total depth frames", len(depth_valid))

    print("rolling bad count?")
    print(np.count_nonzero(definitely_bad))

    diff_threshold = 0.025
    depth_valid_diffs = depth_valid[1:] - depth_valid[:-1]

    diff_bad = np.abs(depth_valid_diffs) > diff_threshold

    print("diff bad count?")
    print(np.count_nonzero(diff_bad))

    mask_out = np.ones_like(depth_valid).astype(bool)
    mask_out[definitely_bad] = False
    mask_out[1:][diff_bad] = False

    return mask_out

# def filter_