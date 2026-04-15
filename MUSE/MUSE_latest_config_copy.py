import argparse
from os import path
from pathlib import Path

import numpy as np
from muse import r_est_jackknife, r_est_naive

from typing import Generator

import glob
import pandas as pd
import re
from scipy.io import wavfile
import toml
import cv2
import math
import os
import tqdm


# Functions to sort file names correctly
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

# experiment_no
experiment_no = 436
samplerate = 125000
camera_calibration_exp = 306
path = f"D:/big_setup/experiment_{experiment_no}/concatenated_data_cam_mic_sync/ssl_data_path/"

# deleting previous MUSE predictions if present
muse_files = glob.glob(path+"idx_*/MUSE*.txt")
for i in muse_files:
    os.remove(i) 




# real world mic positions

# old 8 mic rig positions - accurate version
# mic_positions=np.array([[0.825, 1.14, 0.01],  # units in m
#             [0.815, 0.60, 0.01],
#             [0.825, 0.075, 0.01],
#             [0.44, 0.095, 0.01],
#             [0.055, 0.065, 0.01],  # units in m
#             [0.085, 0.60, 0.01],
#             [0.055, 1.13, 0.01],
#             [0.44, 1.105, 0.01]])

# lowered mic positions - latest config assumes mics are at the surface
mic_positions = np.array([[0.2019, 0.8636, 0.0762],  # units in m
                    [0.9679, 0.8636, 0.0762],
                    [1.1684, 0.7621 , 0.0762],
                    [1.1684, 0.3911 , 0.0762],
                    [1.1684, 0.1871, 0.0762],  # units in m
                    [0.7859 , 0.0, 0.0762],
                    [0.3839, 0.0, 0.0762],
                    [0.0, 0.1031, 0.0762],
                    [0.0, 0.3911, 0.0762],
                    [0.0, 0.6781 , 0.0762]])



# -----------------------------------------------------------------------------------------------------------------
# Sleap calibration of the board has a large error - 10-30 mm
# # I used sleap camera calibration to get the camera matrix 
# with open(f"D:/sleap-3D/session_{camera_calibration_exp}/sleap_calibration.toml", 'r') as f:
#     config = toml.load(f)


# for val in config.keys():
#     if 'cam' in val:
#         if 'gily_center' in config[val]['name']:
#             cameraMatrix = np.array(config[val]['matrix'])
#             distCoeffs = np.array(config[val]['distortions'])

            # sleap's camera rot and translation is not used 
            # rvec = config[val]['rotation'] 
            # tvec = config[val]['translation']

# -----------------------------------------------------------------------------------------------------------------

# manual method of camera calibration
# these values were got by running the code - muse_manual_cam_calibration.py 
# got roughly the same error - can't be removed apparently

cameraMatrix = np.load(f"D:/sleap-3D/session_{camera_calibration_exp}/camera_matrix.npy")
distCoeffs = np.load(f"D:/sleap-3D/session_{camera_calibration_exp}/dist_coeffs.npy")


# rotation and translation matrix from world to camera coordinate system
rvec = np.load(f"D:/sleap-3D/session_{camera_calibration_exp}/cam_rotation.npy")
tvec = np.load(f"D:/sleap-3D/session_{camera_calibration_exp}/cam_translation.npy")


# -----------------------------------------------------------------------------------------------------------------



def audio_gen() -> Generator[list, None, None]:
      global samplerate
      for folder in glob.glob(path+"*"):
        annotations = pd.read_csv(glob.glob(folder+"/*annotations*.csv")[0])
        for index,row in annotations.iterrows():
            if row["channel"] != -1 and row['name'] == "vox": 
                audio_channel_list = glob.glob(folder+"/channel*.wav")
                audio_channel_list.sort(key=natural_keys)
                audio_data = []
                for file in audio_channel_list:
                    samplerate, data = wavfile.read(file)
                    audio_data.append(data)

                audio_data = np.array(audio_data)
                vox_snippet = audio_data[:,int(row["start_seconds"]*samplerate):int(row["stop_seconds"]*samplerate)]
                true_xy = 0
                yield [vox_snippet.T,row["start_seconds"],row["stop_seconds"],folder]





def muse_pred(audio):
    """Run MUSE on audio"""
    # r_est, *_ = r_est_naive(
    #     v=audio.T,
    #     fs=samplerate,
    #     f_lo=2000,
    #     f_hi=samplerate/2,
    #     temp=21,
    #     x_len=1.1684,
    #     y_len=0.8636,
    #     resolution=1e-3,        
    #     mic_positions=mic_positions
    # )

    avg_est, r_ests, _ = r_est_jackknife(
        v=audio.T,
        fs=samplerate,
        f_lo=2000,
        f_hi=samplerate/2,
        temp=21,
        x_len=1.1684,
        y_len=0.8636,
        resolution=1e-3,        
        mic_positions=mic_positions
    )

    r_ests_squeezed = [i.squeeze() for i in r_ests]

    return avg_est.squeeze(), r_ests_squeezed


def run():


        generator = audio_gen()
        for audio,start,stop,folder in tqdm.tqdm(generator):
            # if int(folder.split('_')[-1]) < 91:
            #     continue
            try:
                avg_res_3d, res_3d_pts = muse_pred(audio)
            except Exception as e:
                print(e)
                continue
            avg_res_3d = np.concatenate([avg_res_3d,[0.0]])
            avg_res_2d,_ = cv2.projectPoints(avg_res_3d,
                                 rvec, tvec,
                                 cameraMatrix,
                                 distCoeffs)
            
            res_2d_pts = []
            res_3d_pts_z = []
            for val_3d in res_3d_pts:
                val_3d = np.concatenate([val_3d,[0.0]])
                val_2d,_ = cv2.projectPoints(val_3d,
                                 rvec, tvec,
                                 cameraMatrix,
                                 distCoeffs)
                res_2d_pts.append(val_2d)
                res_3d_pts_z.append(val_3d)
                

    
            results_path = folder + "/MUSE_pred.txt"
            with open(results_path, 'a') as file:
                file.write(str(avg_res_2d)+","+str(avg_res_3d)+","+str(res_2d_pts)+","+str(res_3d_pts_z)+","+str(start)+","+str(stop)+","+folder+"\n")
            #results = [res_2d,res_3d,start,stop,folder]
        
        '''
        # compute error
        half_arena_dims = np.array([0.9144,1.2192]) / 2
        results = (results - half_arena_dims) * 1000
        if "muse_pred" in ctx: 
            del ctx["muse_pred"]
        ctx["muse_pred"] = results

        if "locations" not in ctx:
            print("No true locations found in h5 file")
            return
        true_loc = ctx["locations"][:]
        # true_loc = -true_loc[index]  # dyad and adolescent are flipped
        if len(true_loc.shape) == 3:
            true_loc = true_loc[:, 0, :]

        error = np.linalg.norm(true_loc[index] - results, axis=1)
        error = error / 10
        print(f"Mean error: {error.mean():.2f}cm")
        print(f"Median error: {np.median(error):.2f}cm")
        print(f"Max error: {error.max():.2f}cm")
        print(f"Min error: {error.min():.2f}cm")
        np.savez(
            args.data.with_suffix(".muse_error.npz"),
            error=error,
            index=index,
            true_loc=true_loc,
            preds=results,
        )
        '''


if __name__ == "__main__":
    run()