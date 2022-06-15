from __future__ import print_function
from cProfile import label
import sys
from anyio import current_time
from bleach import clean
import cv2
import os
import numpy as np
import pandas as pd
import datetime
from tomlkit import date
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator


# This script is to calibrate opti-track and azure kinect. It read data from both devices and calibrate them. It does NOT sync frames.
sys.path.insert(1, '../')

sync_dir = os.path.join(os.getcwd(), "sync")
os.makedirs(sync_dir, exist_ok=True)

op_fname = "new_sync_test3.csv"
az_fname = "az_data.csv"

time_break = pd.read_csv("az_time_break.csv", header=None)
az_calib_start_id, az_calib_stop_id, az_scan_start_id, az_scan_stop_id = time_break[0]

# clean az file
az_df = pd.read_csv(az_fname, header=None)
az_df.columns = ['id', 'timestamp', 'threshold', 'position_x', 'position_y', 'position_z']
az_timestamp = az_df['timestamp']
az_df['converted_az_timestamp'] = pd.to_datetime(az_timestamp, format='%Y-%m-%d %H:%M:%S.%f')

### calibration process starts ###

# clean azure kinect dataframe
az_calib_df = az_df[az_df['id'].apply(lambda x : x >= az_calib_start_id and x < az_calib_stop_id)].copy()
az_calib_df['time_delta'] = (az_calib_df["converted_az_timestamp"] - az_calib_df['converted_az_timestamp'].iloc[0]).dt.total_seconds()
az_calib_df = az_calib_df[az_calib_df["position_z"].apply(lambda x : x != 0)]

az_calib_df['2d_distance'] = np.sqrt(az_calib_df['position_x'].astype(float)** 2 + az_calib_df['position_z'].astype(float) ** 2)

# clean original opti-track csv
col_list = [i for i in range(0,10)]
op_df = pd.read_csv(op_fname, usecols=col_list, header=None)
pose_start_time = op_df.iloc[0,9]
cols = ['id', 'timestamp', 'rotation_x', 'rotation_y', 'rotation_z', 'rotation_w', 'position_x', 'position_y', 'position_z']
op_df = op_df.iloc[6:,:9]
op_df.columns = cols

# convert to 24 hour time
if pose_start_time[-2:] == 'PM':
    h = (int)(pose_start_time[11:13]) + 12
    if h >= 24:
        h = h-12
    pose_start_time = pose_start_time[:11] + str(h) + pose_start_time[13:-3]
converted_op_start_time = datetime.datetime.strptime(pose_start_time, '%Y-%m-%d %H.%M.%S.%f')
op_df['converted_op_timestamp'] = converted_op_start_time +  op_df["timestamp"].astype(float).apply(lambda x : datetime.timedelta(seconds=x))


op_calib_df = op_df.iloc[:].copy()
op_calib_df = op_calib_df.dropna()
op_calib_df['2d_distance'] = np.sqrt(op_calib_df['position_x'].astype(float)** 2 + op_calib_df['position_z'].astype(float) ** 2)



az_pos = np.array(az_calib_df['2d_distance']).astype(np.float32)
az_pos_zero_meaned = az_pos - np.mean(az_pos)
az_times = np.array(az_calib_df['time_delta']).astype(np.float32)

op_pos = np.array(op_calib_df['2d_distance']).astype(np.float32)
op_pos_zero_meaned = op_pos - np.mean(op_pos)
op_times = np.array(op_calib_df['timestamp']).astype(np.float32) # op's timestamp is in fact the time delta in op system

op_interp = Akima1DInterpolator(op_times, op_pos)

best_time, best_dist = -1, np.inf

#basically, align az_pos with op_pos at differnet start times by brute force  # aligned to op system delta
#this is rough alignment
for i in range(len(op_times)):
    az_times_shifted = az_times + op_times[i]
    op_interped = op_interp(az_times_shifted)
    op_interped_zero_meaned = op_interped - np.mean(op_interped)

    dist = np.mean(np.square(az_pos_zero_meaned - op_interped_zero_meaned))

    if dist < best_dist:
        best_dist = dist
        best_time = i

print("first pass")
print(best_time, best_dist)

az_times_shifted = az_times + op_times[best_time]

#now, fine refinement with same technique
#finding the best fit within 1 second of rough alignment
best_time_refined, best_dist_refined = -1, np.inf
for i in np.linspace(-0.5, 0.5, 1000):
    az_times_refined = az_times_shifted + i
    op_interped = op_interp(az_times_refined)
    op_interped_zero_meaned = op_interped - np.mean(op_interped)

    dist = np.mean(np.square(az_pos_zero_meaned - op_interped_zero_meaned))

    if dist < best_dist_refined:
        best_dist_refined = dist
        best_time_refined = i

print("refinement pass")
print(best_time_refined, best_dist_refined)

az_times_refined = az_times_shifted + best_time_refined

total_offset = op_times[best_time] + best_time_refined
print("total offset to add to ak timestamps:", total_offset)

az_times_final = az_times + total_offset
op_interpd_final = op_interp(az_times_final)
op_interpd_final -= np.mean(op_interpd_final)

plt.scatter(az_times_final, az_pos_zero_meaned, label="az final")
plt.scatter(az_times_final, op_interpd_final, label="op final")
plt.legend()
plt.savefig(os.path.join(sync_dir, "2d_distance.png"))
plt.clf()



### synchronization process starts ###
az_scan_df = az_df[az_df['id'].apply(lambda x : x >= az_scan_start_id and x < az_scan_stop_id)].copy()
op_scan_df = op_df.copy()

# calculate capture_time_off, sync opti-track to the clock of azure kinect
capture_time_off = total_offset
az_scan_df['time_delta'] = (az_scan_df['converted_az_timestamp'] - az_calib_df['converted_az_timestamp'].iloc[0]).dt.total_seconds() + capture_time_off # relate to the op start time
az_sync_times = np.array(az_scan_df['time_delta']).astype(np.float32)
op_sync_times = np.array(op_scan_df['timestamp']).astype(np.float32)


az_y = np.ones(len(az_sync_times))
op_y = np.ones(len(op_sync_times)) + 1
plt.scatter(az_sync_times, az_y, label="az sync")
plt.scatter(op_sync_times, op_y, label="op sync")
plt.legend()
plt.show()
plt.clf()


new_synced_df = op_scan_df.iloc[0:0].copy()
new_synced_df = new_synced_df.iloc[:,2:-1]

for col in new_synced_df:    
    op_col_data = np.array(op_scan_df[col]).astype(np.float32)
    op_col_interp = Akima1DInterpolator(op_sync_times, op_col_data)
    op_col_interped = op_col_interp(az_sync_times)
    new_synced_df[col] = op_col_interped
    print(new_synced_df)

new_synced_df['Frame'] = az_scan_df['id'].to_list()
cols = new_synced_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
new_synced_df = new_synced_df[cols]
new_synced_df.columns = ['Frame', 'camera_Rotation_X', 'camera_Rotation_Y','camera_Rotation_Z','camera_Rotation_W','camera_Position_X','camera_Position_Y','camera_Position_Z']
new_synced_df.to_csv("new_sync_data.csv")
print(new_synced_df)

### synchronization process ends ###




