# DTTD: Digital-Twin 3D Object Tracking Dataset

## Table of Content
- [Overview](#overview)
- [Requirements](#requirements)
- [Code Structure](#code-structure)
- [Datasets](#datasets)
- [Evaluation](#evaluation)
	- [Evaluation on YCB_Video Dataset](#evaluation-on-ycb_video-dataset)
	- [Evaluation on LineMOD Dataset](#evaluation-on-linemod-dataset)
- [Results](#results)
- [Trained Checkpoints](#trained-checkpoints)
- [Tips for your own dataset](#tips-for-your-own-dataset)
- [Citations](#citations)
- [License](#license)

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
* **calculate_extrinsic**
	* **calculate_extrinsic/aruco_corners.yaml**: 
* **cameras**
* **data_capturing**
* **data_processing**
* **demos**
* **doc**
* **extrinsics_scenes**
* **iphone_app**
* **manual_pose_annotation**
* **models**
* **objects**
* **pose_refinement**
* **quality_control**
* **scene_labeling_generation**
* **scenes**
* **synthetic_data_generation**
* **testing**
* **toolbox**
* **tools**
* **utils** 
	* **datasets/linemod**
		* **datasets/linemod/dataset.py**: Data loader for LineMOD dataset.
		* **datasets/linemod/dataset_config**: 
			* **datasets/linemod/dataset_config/models_info.yml**: Object model info of LineMOD dataset.

## Datasets

This work is tested on two 6D object pose estimation datasets:

* [YCB_Video Dataset](https://rse-lab.cs.washington.edu/projects/posecnn/): Training and Testing sets follow [PoseCNN](https://arxiv.org/abs/1711.00199). The training set includes 80 training videos 0000-0047 & 0060-0091 (choosen by 7 frame as a gap in our training) and synthetic data 000000-079999. The testing set includes 2949 keyframes from 10 testing videos 0048-0059.

* [LineMOD](http://campar.in.tum.de/Main/StefanHinterstoisser): Download the [preprocessed LineMOD dataset](https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7) (including the testing results outputted by the trained vanilla SegNet used for evaluation).

Download YCB_Video Dataset, preprocessed LineMOD dataset and the trained checkpoints (You can modify this script according to your needs.):
```	
./download.sh
```


## Evaluation

### Evaluation on YCB_Video Dataset
For fair comparison, we use the same segmentation results of [PoseCNN](https://rse-lab.cs.washington.edu/projects/posecnn/) and compare with their results after ICP refinement.
Please run:
```
./experiments/scripts/eval_ycb.sh
```
This script will first download the `YCB_Video_toolbox` to the root folder of this repo and test the selected DenseFusion and Iterative Refinement models on the 2949 keyframes of the 10 testing video in YCB_Video Dataset with the same segmentation result of PoseCNN. The result without refinement is stored in `experiments/eval_result/ycb/Densefusion_wo_refine_result` and the refined result is in `experiments/eval_result/ycb/Densefusion_iterative_result`.

After that, you can add the path of `experiments/eval_result/ycb/Densefusion_wo_refine_result/` and `experiments/eval_result/ycb/Densefusion_iterative_result/` to the code `YCB_Video_toolbox/evaluate_poses_keyframe.m` and run it with [MATLAB](https://www.mathworks.com/products/matlab.html). The code `YCB_Video_toolbox/plot_accuracy_keyframe.m` can show you the comparsion plot result. You can easily make it by copying the adapted codes from the `replace_ycb_toolbox/` folder and replace them in the `YCB_Video_toolbox/` folder. But you might still need to change the path of your `YCB_Video Dataset/` in the `globals.m` and copy two result folders(`Densefusion_wo_refine_result/` and `Densefusion_iterative_result/`) to the `YCB_Video_toolbox/` folder.


### Evaluation on LineMOD Dataset
Just run:
```
./experiments/scripts/eval_linemod.sh
```
This script will test the models on the testing set of the LineMOD dataset with the masks outputted by the trained vanilla SegNet model. The result will be printed at the end of the execution and saved as a log in `experiments/eval_result/linemod/`.


## Results

* YCB_Video Dataset:

Quantitative evaluation result with ADD-S metric compared to other RGB-D methods. `Ours(per-pixel)` is the result of the DenseFusion model without refinement and `Ours(iterative)` is the result with iterative refinement.

<p align="center">
	<img src ="assets/result_ycb.png" width="600" />
</p>

**Important!** Before you use these numbers to compare with your methods, please make sure one important issus: One difficulty for testing on the YCB_Video Dataset is how to let the network to tell the difference between the object `051_large_clamp` and `052_extra_large_clamp`. The result of all the approaches in this table uses the same segmentation masks released by PoseCNN without any detection priors, so all of them suffer a performance drop on these two objects because of the poor detection result and this drop is also added to the final overall score. If you have added detection priors to your detector to distinguish these two objects, please clarify or do not copy the overall score for comparsion experiments.

* LineMOD Dataset:

Quantitative evaluation result with ADD metric for non-symmetry objects and ADD-S for symmetry objects(eggbox, glue) compared to other RGB-D methods. High performance RGB methods are also listed for reference.

<p align="center">
	<img src ="assets/result_linemod.png" width="500" />
</p>

The qualitative result on the YCB_Video dataset.

<p align="center">
	<img src ="assets/compare.png" width="600" />
</p>

## Trained Checkpoints
You can download the trained DenseFusion and Iterative Refinement checkpoints of both datasets from [Link](https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7).

## Tips for your own dataset
As you can see in this repo, the network code and the hyperparameters (lr and w) remain the same for both datasets. Which means you might not need to adjust too much on the network structure and hyperparameters when you use this repo on your own dataset. Please make sure that the distance metric in your dataset should be converted to meter, otherwise the hyperparameter w need to be adjusted. Several useful tools including [LabelFusion](https://github.com/RobotLocomotion/LabelFusion) and [sixd_toolkit](https://github.com/thodan/sixd_toolkit) has been tested to work well. (Please make sure to turn on the depth image collection in LabelFusion when you use it.)


## Citations
Please cite [DenseFusion](https://sites.google.com/view/densefusion) if you use this repository in your publications:
```
@article{wang2019densefusion,
  title={DenseFusion: 6D Object Pose Estimation by Iterative Dense Fusion},
  author={Wang, Chen and Xu, Danfei and Zhu, Yuke and Mart{\'\i}n-Mart{\'\i}n, Roberto and Lu, Cewu and Fei-Fei, Li and Savarese, Silvio},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```

## License
Licensed under the [MIT License](LICENSE)


# Using OptiTrack output to generate GT semantic segmentation

Basically, we take the optitrack which tracks markers, and we need to render the models in those poses, and then collect GT semantic segmentation data like that.

## Objects and Scenes data:
https://docs.google.com/spreadsheets/d/1weyPvCyxU82EIokMlGhlK5uEbNt9b8a-54ziPpeGjRo/edit?usp=sharing

Final dataset output:
 * `objects` folder
 * `cameras` folder
 * `scenes` folder certain data:
 	 * `scenes/<scene number>/data/` folder
 	 * `scenes/<scene number>/scene_meta.yaml` metadata
 * `toolbox` folder

# How to run
 1. Setup
	 1. Place ARUCO marker near origin (doesn't actually matter where it is anymore, but makes sense to be near OptiTrack origin)
	 2. Calibrate Opti (if you want, don't need to do this everytime, or else extrinsic changes)
	 3. Place markers on the corners of the aruco marker, use this to compute the aruco -> opti transform
	 	 * Place marker positions into `calculate_extrinsic/aruco_corners.yaml`
 2. Record Data (`tools/capture_data.py`)
     1. ARUCO Calibration
	 2. Data collection
	 	 * If extrinsic scene, data collection phase should be spent observing ARUCO marker
	 3. Example: <code>python tools/capture_data.py --scene_name official_test az_camera</code>
		 
 3. Check if the Extrinsic file exists
	 1. If Extrinsic file doesn't exist, then you need to calculate Extrinsic through Step 4
	 2. Otherwise, process data through Step 5 to generate groundtruth labels
 4. Process iPhone Data (if iPhone Data)
	1. Convert iPhone data formats to Kinect data formats (`tools/process_iphone_data.py`)
		* This tool converts everything to common image names, formats, and does distortion parameter fitting
	2. Continue with step 5 or 6 depending on whether computing an extrinsic or capturing scene data
 5. Process Extrinsic Data to Calculate Extrinsic (If extrinsic scene)
	 1. Clean raw opti poses (`tools/process_data.py`) 
	 2. Sync opti poses with frames (`tools/process_data.py`)
	 3. Calculate camera extrinsic (`tools/calculate_camera_extrinsic.py`)
 6. Process Data (If data scene)
	 1. Clean raw opti poses (`tools/process_data.py`) <br>
	 Example: <code>python tools/process_data.py --scene_name [SCENE_NAME]</code>
	 2. Sync opti poses with frames (`tools/process_data.py`) <br>
	 Example: <code>python tools/process_data.py --scene_name [SCENE_NAME]</code>
	 3. Manually annotate first frame object poses (`tools/manual_annotate_poses.py`)
	 	 1. Modify (`[SCENE_NAME]/scene_meta.yml`) by adding (`objects`) field to the file according to objects and their corresponding ids.<br>
			Example: <code>python tools/manual_annotate_poses.py official_test</code>
	 4. Recover all frame object poses and verify correctness (`tools/generate_scene_labeling.py`) <br>
	 Example: <code>python tools/generate_scene_labeling.py --fast [SCENE_NAME]</code>
	 5. Generate semantic labeling (`tools/generate_scene_labeling.py`)<br>
	 Example: <code>python /tools/generate_scene_labeling.py [SCENE_NAME]</code>
	 6. Generate per frame object poses (`tools/generate_scene_labeling.py`)<br>
	 Example: <code>python tools/generate_scene_labeling.py [SCENE_NAME]</code>


# Minutia
 * Extrinsic scenes have their color images inside of `data` stored as `png`. This is to maximize performance. Data scenes have their color images inside of `data` stored as `jpg`. This is necessary so the dataset remains usable.
 * iPhone spits out `jpg` raw color images, while Azure Kinect skips out `png` raw color images.

# Best Scene Collection Practices
 * Good synchronization phase by observing ARUCO marker, for Azure Kinect keep in mind interference from OptiTrack system.
 * Don't have objects that are in our datasets in the background. Make sure they are out of view!
 * Minimize number of extraneous ARUCO markers/APRIL tags that appear in the scene.
 * Stay in the yellow area for best OptiTrack tracking.
 * Move other cameras out of area when collecting data to avoid OptiTrack confusion.
 * Run `manual_annotate_poses.py` on all scenes after collection in order to archive extrinsic.
 * We want to keep the data anonymized. Avoid school logos and members of the lab appearing in frame.
 * Perform 90-180 revolution around objects, one way. Try to minimize stand-still time.


# Todo: Model Evaluation for this dataset
Select SOTA pose estimation & image segmentation models and perform evaluations according to certain metrics.


# Using OptiTrack output to generate GT semantic segmentation

Basically, we take the optitrack which tracks markers, and we need to render the models in those poses, and then collect GT semantic segmentation data like that.

## Objects and Scenes data:
https://docs.google.com/spreadsheets/d/1weyPvCyxU82EIokMlGhlK5uEbNt9b8a-54ziPpeGjRo/edit?usp=sharing

Final dataset output:
 * `objects` folder
 * `cameras` folder
 * `scenes` folder certain data:
 	 * `scenes/<scene number>/data/` folder
 	 * `scenes/<scene number>/scene_meta.yaml` metadata
 * `toolbox` folder

# How to run
 1. Setup
	 1. Place ARUCO marker near origin (doesn't actually matter where it is anymore, but makes sense to be near OptiTrack origin)
	 2. Calibrate Opti (if you want, don't need to do this everytime, or else extrinsic changes)
	 3. Place markers on the corners of the aruco marker, use this to compute the aruco -> opti transform
	 	 * Place marker positions into `calculate_extrinsic/aruco_corners.yaml`
 2. Record Data (`tools/capture_data.py`)
     1. ARUCO Calibration
	 2. Data collection
	 	 * If extrinsic scene, data collection phase should be spent observing ARUCO marker
	 3. Example: <code>python tools/capture_data.py --scene_name official_test az_camera</code>
		 
 3. Check if the Extrinsic file exists
	 1. If Extrinsic file doesn't exist, then you need to calculate Extrinsic through Step 4
	 2. Otherwise, process data through Step 5 to generate groundtruth labels
 4. Process iPhone Data (if iPhone Data)
	1. Convert iPhone data formats to Kinect data formats (`tools/process_iphone_data.py`)
		* This tool converts everything to common image names, formats, and does distortion parameter fitting
	2. Continue with step 5 or 6 depending on whether computing an extrinsic or capturing scene data
 5. Process Extrinsic Data to Calculate Extrinsic (If extrinsic scene)
	 1. Clean raw opti poses (`tools/process_data.py`) 
	 2. Sync opti poses with frames (`tools/process_data.py`)
	 3. Calculate camera extrinsic (`tools/calculate_camera_extrinsic.py`)
 6. Process Data (If data scene)
	 1. Clean raw opti poses (`tools/process_data.py`) <br>
	 Example: <code>python tools/process_data.py --scene_name [SCENE_NAME]</code>
	 2. Sync opti poses with frames (`tools/process_data.py`) <br>
	 Example: <code>python tools/process_data.py --scene_name [SCENE_NAME]</code>
	 3. Manually annotate first frame object poses (`tools/manual_annotate_poses.py`)
	 	 1. Modify (`[SCENE_NAME]/scene_meta.yml`) by adding (`objects`) field to the file according to objects and their corresponding ids.<br>
			Example: <code>python tools/manual_annotate_poses.py official_test</code>
	 4. Recover all frame object poses and verify correctness (`tools/generate_scene_labeling.py`) <br>
	 Example: <code>python tools/generate_scene_labeling.py --fast [SCENE_NAME]</code>
	 5. Generate semantic labeling (`tools/generate_scene_labeling.py`)<br>
	 Example: <code>python /tools/generate_scene_labeling.py [SCENE_NAME]</code>
	 6. Generate per frame object poses (`tools/generate_scene_labeling.py`)<br>
	 Example: <code>python tools/generate_scene_labeling.py [SCENE_NAME]</code>


# Minutia
 * Extrinsic scenes have their color images inside of `data` stored as `png`. This is to maximize performance. Data scenes have their color images inside of `data` stored as `jpg`. This is necessary so the dataset remains usable.
 * iPhone spits out `jpg` raw color images, while Azure Kinect skips out `png` raw color images.

# Best Scene Collection Practices
 * Good synchronization phase by observing ARUCO marker, for Azure Kinect keep in mind interference from OptiTrack system.
 * Don't have objects that are in our datasets in the background. Make sure they are out of view!
 * Minimize number of extraneous ARUCO markers/APRIL tags that appear in the scene.
 * Stay in the yellow area for best OptiTrack tracking.
 * Move other cameras out of area when collecting data to avoid OptiTrack confusion.
 * Run `manual_annotate_poses.py` on all scenes after collection in order to archive extrinsic.
 * We want to keep the data anonymized. Avoid school logos and members of the lab appearing in frame.
 * Perform 90-180 revolution around objects, one way. Try to minimize stand-still time.


# Todo: Model Evaluation for this dataset
Select SOTA pose estimation & image segmentation models and perform evaluations according to certain metrics.
