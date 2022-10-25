import os
import sys
import json
import numpy as np
from data_process import *
from coord_transform import Point2d

# Locating OpenPose json output files
root_dict = os.path.dirname(os.path.realpath(__file__))
if len(sys.argv) == 3:
    json_dir = os.path.join(root_dict, sys.argv[1])
else:
    print("Problem with number of arguments! Usage: 'python preparing_2Djsons.py <2Djson_dirname> <clean2Djson_dirname>'")
    exit()

try:
    files = os.listdir(json_dir)
    files.sort()
except:
    print("Json files don't exist!")
    exit()

detconf_tresh = 0.5
NPOINTS = 25

# Loading OpenPose output for data cleaning
for f_idx in range(len(files)):
    print(files[f_idx])
    file = open(os.path.join(json_dir,files[f_idx]))
    jsonData = json.load(file)
    file.close() 
    Data_curr = np.expand_dims(np.zeros(NPOINTS, dtype=object), axis=1) # Helper table for loading current frame data
    Conf_curr = np.expand_dims(np.zeros(NPOINTS, dtype=object), axis=1) # Helper table to save detection confidence

    # If data for current frame is missing:
    if len(jsonData['people']) == 0:
        for i in range(NPOINTS):
            Data_curr[i][0] = Point2d(0, 0)
            Conf_curr[i][0] = 0
    else:
        for i in range(0, NPOINTS * 3, 3):
            # Checks if detection confidence is higher than detconf_tresh
            if jsonData['people'][0]['pose_keypoints_2d'][i + 2] > detconf_tresh:
                Data_curr[int(i/3)][0] = Point2d(jsonData['people'][0]['pose_keypoints_2d'][i],
                                    jsonData['people'][0]['pose_keypoints_2d'][i + 1])
                Conf_curr[int(i/3)][0] = jsonData['people'][0]['pose_keypoints_2d'][i + 2]
            else:
                Data_curr[int(i/3)][0] = Point2d(0, 0)
                Conf_curr[int(i/3)][0] = jsonData['people'][0]['pose_keypoints_2d'][i + 2]
    
    # Extends Data table, or creates it if first frame
    if f_idx == 0:
        Data = Data_curr
        Conf = Conf_curr
    else:
        Data = np.concatenate((Data, Data_curr), axis=1)
        Conf = np.concatenate((Conf, Conf_curr), axis=1)

# Data cleaning
Data = resolve_zeros(Data) # Imputing missing data 
Data = savgol_filter(Data, 5, 3) # Applying Savitzky-Golay filter

Data = np.transpose(Data)
Conf = np.transpose(Conf)

# Making new folder
new_dir = os.path.join(root_dict, sys.argv[2])
if not os.path.exists(new_dir):
    os.mkdir(new_dir)

# Saving cleaned jsons - ready for 3d lifting - to new folder
for idx, (frame_data, frame_conf) in enumerate(zip(Data, Conf)):
    jsonFile = sys.argv[2]+ '_' + str(idx).zfill(12) + '_keypoints.json'
    newjsonFile = open(os.path.join(new_dir, jsonFile), "w")
    points = []
    for point_data, conf_data in zip(frame_data, frame_conf):
        points.append(point_data.x)
        points.append(point_data.y)
        points.append(conf_data)
    string = ", ".join(str(x) for x in points)
    string = {'version': 1.2, 'people': [{'pose_keypoints_2d': points, 'face_keypoints_2d': [], 'hand_left_keypoints_2d': [], 'hand_right_keypoints_2d': [], 'pose_keypoints_3d': [], 'face_keypoints_3d': [], 'hand_left_keypoints_3d': [], 'hand_right_keypoints_3d': []}]}
    newjsonFile.write(json.dumps(string))
    newjsonFile.close()

