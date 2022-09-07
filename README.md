# DTTD: Digital-Twin Tracking Dataset Official Repository

## Overview

This repository is the implementation code of the paper "Digital-Twin Tracking Dataset (DTTD): A Time-of-Flight 3D Object Tracking Dataset for High-Quality AR Applications".

In this work we create a novel RGB-D dataset, Digital-Twin Tracking Dataset (DTTD), to enable further research of the digital-twin tracking problem in pursuit of a Digital Twin solution. In our dataset, we select two time-of-flight (ToF) depth sensors, Microsoft Azure Kinect and Apple iPhone 12 Pro, to record 100 scenes each of 16 common purchasable objects, each frame annotated with a per-pixel semantic segmentation and ground truth object poses. We also provide source code in this repository as references to data generation and annotation pipeline in our paper. 

Link for Dataset (To be released)


## Requirements

Before running our data generation and annotation pipeline, you can activate a __conda__ environment where Python Version >= 3.7:
```
conda create --name [YOUR ENVIR NAME] python = [PYTHON VERSION]
conda activate [YOUR ENVIR NAME]
```

then install all necessary packages:
```
pip install -r requirements.txt
```

## Code Structure
* **calculate_extrinsic**: extrinsic information
* **cameras**: camera information
* **data_capturing**: helper package for data capturing
* **data_processing**: helper package for data processing
* **demos**: demo videos
* **doc**: demo images
* **extrinsics_scenes**: folder to save all extrinsic scenes
* **iphone_app**: iPhone app development for capturing RGBD images for iPhone 12 Pro camera
* **manual_pose_annotation**: helper package for pose annotation
* **models**: baseline deep learning 6D pose estimation algorithms
* **objects**: object models that we use in DTTD (with corresponding scale and texture)
* **pose_refinement**: helper package for pose refinement
* **quality_control**: helper package for reviewing manual annotations
* **scene_labeling_generation**: helper package for generating labels
* **scenes**: folder to save all recorded RGBD data
* **synthetic_data_generation**: helper package for generating synthetic data
* **testing**: package to test aruco marker's appearance, extrinsic's validity, etc.
* **toolbox**: package to generate data for model training
* **tools**: commands for running the pipelines. Details in **tools/README.md**.
* **utils**: utils package

## Dataset Structure

Final dataset output:
 * `objects` folder
 * `scenes` folder certain data:
 	 * `scenes/<scene name>/data/` folder
 	 * `scenes/<scene name>/scene_meta.yaml` metadata
 * `toolbox` folder

## What you Need to Collect your own Data
 1. OptiTrack Motion Capture system with Motive tracking software
	* This doesn't have to be running on the same computer as the other sensors. We will export the tracked poses to a CSV file.
	* Create a rigid body to track a camera's OptiTrack markers, give the rigid body the same name that is passed into `tools/capture_data.py`
 2. Microsoft Azure Kinect
	* We interface with the camera using Microsoft's K4A SDK: https://github.com/microsoft/Azure-Kinect-Sensor-SDK
 3. iPhone 12 Pro / iPhone 13
	* Please build the project in `iphone_app/` in XCode and install on the mobile device.

## Data Collection Pipeline
### Configuration & Setup
  1. Place ARUCO marker somewhere visible
  2. Place markers on the corners of the aruco marker, we use this to compute the (aruco -> opti) transform
  3. Place marker positions into `calculate_extrinsic/aruco_corners.yaml`, labeled under keys: `quad1`, `quad2`, `quad3`, and `quad4`.

### Record Data (`tools/capture_data.py`)
  1. Data collection
      * If extrinsic scene, data collection phase should be spent observing ARUCO marker, run `tools/capture_data.py --extrinsic`
  2. Example data collection scene (not extrinsic): `python tools/capture_data.py --scene_name test az_camera1`

#### Data Recording Process
  1. Start the OptiTrack recording
  2. Synchronization Phase
	  1. Press `c` to begin recording data
	  2. Observe the ARUCO marker in the scene and move the camera in different trajectories to build synchronization data
	  3. Press `p` when finished
  3. Data Capturing Phase
      1. Press `d` to begin recording data
	  2. If extrinsic scene, observe the ARUCO marker.
	  3. If data collection scene, observe objects to track
	  4. Press `q` when finished
  4. Stop OptiTrack recording
  5. Export OptiTrack recording to a CSV file with 60Hz report rate.
  6. Move tracking CSV file to `<scene name>/camera_poses/camera_pose.csv`
		 
### Process iPhone Data (if iPhone Data)
  1. Convert iPhone data formats to Kinect data formats (`tools/process_iphone_data.py`)
		* This tool converts everything to common image names, formats, and does distortion parameter fitting
  2. Continue with step 5 or 6 depending on whether computing an extrinsic or capturing scene data

### Process Extrinsic Data to Calculate Extrinsic (If extrinsic scene)
  1. Clean raw opti poses (`tools/process_data.py --extrinsic`) 
  2. Sync opti poses with frames (`tools/process_data.py --extrinsic`)
  3. Calculate camera extrinsic (`tools/calculate_camera_extrinsic.py`)
  4. Output will be placed in `cameras/<camera name>/extrinsic.txt`

### Process Data (If data scene)
  1. Clean raw opti poses (`tools/process_data.py`) <br>
	 Example: <code>python tools/process_data.py --scene_name [SCENE_NAME]</code>
  2. Sync opti poses with frames (`tools/process_data.py`) <br>
	 Example: <code>python tools/process_data.py --scene_name [SCENE_NAME]</code>
  3. Manually annotate first frame object poses (`tools/manual_annotate_poses.py`)
	 	 1. Modify (`[SCENE_NAME]/scene_meta.yml`) by adding (`objects`) field to the file according to objects and their corresponding ids.<br>
			Example: `python tools/manual_annotate_poses.py test`
  4. Recover all frame object poses and verify correctness (`tools/generate_scene_labeling.py`) <br>
	 Example: <code>python tools/generate_scene_labeling.py --fast [SCENE_NAME]</code>
	 1. Generate semantic labeling (`tools/generate_scene_labeling.py`)<br>
	 Example: <code>python /tools/generate_scene_labeling.py [SCENE_NAME]</code>
	 2. Generate per frame object poses (`tools/generate_scene_labeling.py`)<br>
	 Example: <code>python tools/generate_scene_labeling.py [SCENE_NAME]</code>

## Minutia
 * Extrinsic scenes have their color images inside of `data` stored as `png`. This is to maximize performance. Data scenes have their color images inside of `data` stored as `jpg`. This is necessary so the dataset remains usable.
 * iPhone spits out `jpg` raw color images, while Azure Kinect skips out `png` raw color images.

## Best Scene Collection Practices
 * Good synchronization phase by observing ARUCO marker, for Azure Kinect keep in mind interference from OptiTrack system.
 * Don't have objects that are in our datasets in the background. Make sure they are out of view!
 * Minimize number of extraneous ARUCO markers/APRIL tags that appear in the scene.
 * Stay in the yellow area for best OptiTrack tracking.
 * Move other cameras out of area when collecting data to avoid OptiTrack confusion.
 * Run `manual_annotate_poses.py` on all scenes after collection in order to archive extrinsic.
 * We want to keep the data anonymized. Avoid school logos and members of the lab appearing in frame.
 * Perform 90-180 revolution around objects, one way. Try to minimize stand-still time.
 
