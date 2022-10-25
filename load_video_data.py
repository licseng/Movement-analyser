import numpy as np
import os, sys 
import json 
from coord_transform import *
import cv2
from constants import NPOINTS, PLIST_2D, PLIST_3D
from graphics import *


def loading_2dvdata(json_dict, Video):
    """
    Loads 2D data from json files extracted from video.
    Opens video for script testing.
    """
    starting_frame = 900
    curr_frame = starting_frame
    detconf_tresh = 0.5

    Data_curr = np.expand_dims(np.zeros(NPOINTS, dtype=object), axis=1)  # Helper table for loading current frame data
    while Video.isOpened():
        print(curr_frame)
        _, frame = Video.read()
        file_ref = open(os.path.join(json_dict, sys.argv[2])+ '_' + str(curr_frame).zfill(12) + '_keypoints.json', "r")
        jsonData = json.load(file_ref)

        # Checks if frame has a detected person on it 
        if len(jsonData['people']) == 0:
            for i in range(NPOINTS):
                Data_curr[i][0] = Point2d(0, 0)
        else:
            B_transf, origo, ref_bottom = trunk2d(jsonData)
            for i in range(NPOINTS):
                if all(item != 0 for item in ref_bottom) and all(item != 0 for item in origo):
                    # Checks if detection confidence is higher than detconf_tresh
                    if jsonData['people'][0]['pose_keypoints_2d'][PLIST_2D[i] + 2] > detconf_tresh:
                        Data_curr[i][0] = transform(B_transf, np.array([jsonData['people'][0]['pose_keypoints_2d'][PLIST_2D[i]],
                                                            jsonData['people'][0]['pose_keypoints_2d'][PLIST_2D[i] + 1]]), origo, mode = "2d")
                    else:
                        Data_curr[i][0] = Point2d(0, 0)
                else:
                    Data_curr[i][0] = Point2d(0, 0)

        # Extends Data table, or creates it if first frame
        if curr_frame == starting_frame:
            #Data = np.expand_dims(np.zeros(NPOINTS, dtype=object),axis=1)
            #for i in range(NPOINTS):
            Data = Data_curr
        else:
            Data = np.concatenate((Data, Data_curr), axis=1)

        # Draws helper line on video for testing purposes
        if all(item != 0 for item in ref_bottom) and all(item != 0 for item in origo):
            cv2.line(frame, (int(ref_bottom[0]), int(ref_bottom[1])), (int(origo[0]), int(origo[1])), (0, 0, 255), 3)
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

        curr_frame += 1

    Video.release()
    cv2.destroyAllWindows()

    return Data
    
def project_stickfig(body):
    """
    Projects 3D data to 2D graphics for testing purposes during data loading.
    """
    win = GraphWin(width=1000, height=1000)

    for i in range(int(len(body)/3)):
        pt = Circle(center=Point(np.round(body[3 * i]), np.round(body[3 * i + 1])), radius=5)
        print(np.round(body[i * 3]), np.round(body[i * 3 + 1]))
        pt.draw(win)

    win.getMouse()
    win.close()

def loading_3dvdata(jsonData):
    """
    Loads 3D data from json file.
    """
    Data_curr = np.expand_dims(np.zeros(NPOINTS, dtype=object), axis =1)
    for i in range(len(jsonData)):
        B_transf, origo = trunk3d(jsonData, i)
        print(i)
        for j in range(NPOINTS):
            Data_curr[j][0] = transform(B_transf, np.array([jsonData[str(i)][str(PLIST_3D[j])]["translate"][0],
                              jsonData[str(i)][str(PLIST_3D[j])]["translate"][1], jsonData[str(i)][str(PLIST_3D[j])]["translate"][2]]), origo, mode = "3d")
        if i == 0:
            Data = Data_curr
            #project_stickfig(Data_curr)
        else:
            Data = np.concatenate((Data, Data_curr), axis=1)
    
    return Data

