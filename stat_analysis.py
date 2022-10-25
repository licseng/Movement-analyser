# stat_analysis.py
"""
Statistical analysis script for movement Data extracted in json format with OpenPose (2d) or 3d-pose-baseline
OpenPose or 3d-pose-baseline needs to be run externally, this script works with their output files.
In 2d mode:
    Usage: 'python stat_analysis.py 2d <2Djson_dirname> <video_name>'
    Script works with the output directory of OpenPose which contains json files for every video frame with 25 body points
    Every point has 3 variables: 1. x coordinate, 2. y coordinate, 3. detection confidence
In 3d mode:
    Usage: 'python stat_analysis.py 3d <3Djson_filename>'
    Script works with the output json file of 3d-pose-baseline, a pretrained model that lifts OpenPose's output 2d coordinates to 3d space.
    However the lifting algorithm accepts only cleaned json files, so 'python preparing_2Djsons.py <2Djson_dirname> <clean2Djson_dirname>' has to be run in advance!
    Every 3d-pose-baseline output point has 3 variables: 1. x coordinate, 3. y coordinate, 3. z coordinate

Mind that:
- the extracted statistics are NOT validated properly! More smoothing algorithms (for time series data) should be tested.
- during comparison of videos' statistics, fps of the examined videos matter! 
- with different versions of OpenPose and 3d-pose-baseline the examined bodypoints and their index could change! 
"""
import sys
import os
import numpy as np
import cv2
from load_video_data import *
from coord_transform import *
from data_process import *
from metrics import *
import pandas as pd


if  sys.argv[1] == "2d":
    # Locating json and video files 
    root_dict = os.path.dirname(os.path.realpath(__file__))
    if len(sys.argv) == 4:
        json_dir = os.path.join(root_dict, sys.argv[2])
        video_file = os.path.join(root_dict, sys.argv[3])
    elif len(sys.argv) != 4:
        print("Problem with arguments! Usage: 'python stat_analysis.py <mode> <2Djson_dirname> <video_name>'")
        exit()

    try:
        files = os.listdir(json_dir)
        files.sort()
    except:
        print("Json files don't exist!")
        exit()

    # Opening videos first caption
    Video = cv2.VideoCapture(video_file)

    if not Video.isOpened():
        print("Video doesn't exist!")
        exit()
    
    print("Video fps: ", int(Video.get(cv2.CAP_PROP_FPS)))

    # Loading data from json files and opening video for test purposes
    Data = loading_2dvdata(json_dir, Video)
    
    # Data cleaning
    Data = resolve_zeros(Data) # Imputing missing data 
    Data = savgol_filter(Data, 5, 3) # Applying Savitzky-Golay filter

elif sys.argv[1] == "3d":
    root_dict = os.path.dirname(os.path.realpath(__file__))
    if len(sys.argv) == 3:
        json_file = os.path.join(root_dict, sys.argv[2])
    elif len(sys.argv) != 3:
        print("Problem with arguments! Usage: 'python stat_analysis.py <mode> <3Djson_filename>'")
        exit()

    try:
        openJson = open(json_file)
    except:
        print("Json file with 3D lifted coordinates doesn't exit!")
        exit()

    jsonData = json.load(openJson)
    Data = loading_3dvdata(jsonData)

    # Data cleaning is already done before 3D lifting

# Metrics
window_size = 5   
Disp_raw, signedDisp_raw = displacement(Data, mode = sys.argv[1]) # Calculating displacement between neighbouring frames
Disp_win_avgs, disp_max = displacement_window(Disp_raw, window_size, 0.2)
Disp_win_signed_avgs, _ = displacement_window(signedDisp_raw, window_size, 0.2, True)
disp_sum = displacement_sum(Disp_win_avgs)
disp_var = displacement_var(Disp_win_avgs)
velocity_avg = disp_sum / np.shape(Data)[1]
scalars_avg, scalars_var = scalars(Data)

# Putting metrics in dataframe
Columns = ["relbow", "rhand", "lelbow", "lhand", "rknee", "rfoot", "lknee", "lfoot", "head", "rshoulder", "lshoulder"]
Rows = ["Distance taken", "Max Displacement", "Avg Velocity", "Var Displacement", "Avg Scalars", "Var Scalars"]
df = pd.DataFrame([disp_sum, disp_max, velocity_avg, disp_var, scalars_avg, scalars_var], Rows, Columns)
# print(df)

# Saving dataframe to csv
df.to_csv("metrics" + sys.argv[1] + ".csv")

plot_data(Disp_win_signed_avgs, sys.argv[1])