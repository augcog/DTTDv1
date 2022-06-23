import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))


class OptiARIMASmoother():
    @staticmethod
    def visualize_result(res, data, model='arima'):
        print(res.summary())
        
        plt.plot(res.resid)
        plt.title('Training Residuals')
        plt.xlabel('Months')
        plt.ylabel('Temperature in C')
        plt.show()

        print('Mean squared training error: %s' % str(np.square(res.resid).mean()))

        model_fit = res.predict()

        plt.plot(model_fit, label='Prediction')
        plt.plot(data, label='Real values')
        plt.legend()
        plt.title('Model fit on training data')
        plt.xlabel('Months')
        plt.ylabel('Temperature in C')
        plt.show()
        
        #TODO: Calculate the mean squared testing error
        squared_testing_error = np.square(model_fit - data).mean()
        print('Mean squared testing error: %s' % str(squared_testing_error))
    
    @staticmethod
    def smooth_opti_poses_arima(scene_dir, pose_df, write_smoothed_to_file=False):
        
        pose_df = pose_df.replace('', np.NaN).astype(np.float32)
        pose_df = pose_df.interpolate()

        xyzs = np.array(pose_df[['camera_Position_X', 'camera_Position_Y', 'camera_Position_Z']]).astype(np.float32)
        rots = np.array(pose_df[['camera_Rotation_X', 'camera_Rotation_Y', 'camera_Rotation_Z', 'camera_Rotation_W']]).astype(np.float32)

        x = xyzs[:,0]
        y = xyzs[:,1]
        z = xyzs[:,2]
        a = rots[:,0]
        b = rots[:,1]
        c = rots[:,2]
        d = rots[:,3]

        op = 2
        od = 0
        oq = 2

        res_x = ARIMA(x, order=(op, od, oq)).fit().predict()
        res_y = ARIMA(y, order=(op, od, oq)).fit().predict()
        res_z = ARIMA(z, order=(op, od, oq)).fit().predict()
        res_a = ARIMA(a, order=(op, od, oq)).fit().predict()
        res_b = ARIMA(b, order=(op, od, oq)).fit().predict()
        res_c = ARIMA(c, order=(op, od, oq)).fit().predict()
        res_d = ARIMA(d, order=(op, od, oq)).fit().predict()

        smoothed_pose_df = pose_df.copy()
        smoothed_pose_df['camera_Position_X'] = res_x
        smoothed_pose_df['camera_Position_Y'] = res_y
        smoothed_pose_df['camera_Position_Z'] = res_z
        smoothed_pose_df['camera_Rotation_X'] = res_a
        smoothed_pose_df['camera_Rotation_Y'] = res_b
        smoothed_pose_df['camera_Rotation_Z'] = res_c
        smoothed_pose_df['camera_Rotation_W'] = res_d

        if write_smoothed_to_file:
            output_file = os.path.join(scene_dir, 'camera_poses', 'camera_poses_smoothed.csv')
            smoothed_pose_df.to_csv(output_file)

        return smoothed_pose_df