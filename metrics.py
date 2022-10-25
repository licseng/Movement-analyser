from os import preadv
import numpy as np
from constants import NPOINTS
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt


def displacement(Data, mode = "2d"): 
    """
    Returns two tables, one with the Eucledian distance, another with signed distance of neighbouring frame points.
    Can be used in "2d" and "3d" mode.
    """
    data_length = np.shape(Data)[1] 
    Diffs = np.empty((NPOINTS, data_length - 1))
    signedDiffs = np.empty((NPOINTS, data_length - 1))
    for body_point in range(NPOINTS):
        for frame in range(1, data_length - 1):
            if mode == "2d":
                dist = np.sqrt((Data[body_point][frame].x - Data[body_point][frame - 1].x)**2 + (Data[body_point][frame].y - Data[body_point][frame - 1].y)**2)
                signed_dist = (Data[body_point][frame].x - Data[body_point][frame - 1].x) + (Data[body_point][frame].y - Data[body_point][frame - 1].y)
            elif mode == "3d":
                dist = ((Data[body_point][frame].x - Data[body_point][frame - 1].x)**2 + (Data[body_point][frame].y - Data[body_point][frame - 1].y)**2 + (Data[body_point][frame].z - Data[body_point][frame - 1].z)**2) ** (1./3.)
                signed_dist = (Data[body_point][frame].x - Data[body_point][frame - 1].x) + (Data[body_point][frame].y - Data[body_point][frame - 1].y + (Data[body_point][frame].z - Data[body_point][frame - 1].z))    
            if dist > 0.0001:
                Diffs[body_point][frame] = dist
                signedDiffs[body_point][frame] = signed_dist
            else:
                Diffs[body_point][frame] = 0
                signedDiffs[body_point][frame] = 0

    return Diffs, signedDiffs

def displacement_window(Disp, windowsize, treshold = 0, exp = False):
    """
    Average displacement of 'windowsize' number of neighbouring displacements OR exponential smoothing on data
    Additional noise reduction can be exponential smoothing and average smoothing.
    """
    length = np.shape(Disp)[1]
    Diffs_window = np.empty((NPOINTS, length))

    if exp == True:
        print("exponential smoothing")
        for body_point in range(NPOINTS):
            alpha = 0.5
            for frame in range(length):
                if frame == 0:
                    prev = Disp[body_point][frame]
                    Diffs_window[body_point][frame] = prev
                elif frame < 5:
                    Diffs_window[body_point][frame] = alpha * np.average(Disp[body_point][:frame]) + (1 - alpha) * prev
                    prev = Diffs_window[body_point][frame]
                else:
                    Diffs_window[body_point][frame] = alpha * np.average(Disp[body_point][frame - windowsize : frame]) + (1 - alpha) * prev
                    prev = Diffs_window[body_point][frame]
    else:    
        print("average smoothing")    
        for body_point in range(NPOINTS):
            for frame in range(length):
                if frame == 0:
                    Diffs_window[body_point][frame] = Disp[body_point][frame]
                elif frame < 5:
                    Diffs_window[body_point][frame] = np.average(Disp[body_point][:frame])
                else:
                    Diffs_window[body_point][frame] = np.average(Disp[body_point][frame - windowsize : frame])
    
    Diffs_min = np.min(np.abs(Diffs_window), axis=1)
    Diffs_max = np.max(np.abs(Diffs_window), axis=1)

    # Bottom 'treshold' percantage of displacement is set to zero due to assumed detection error
    if treshold > 0:
        for body_point in range(NPOINTS):
            Diffs_window[body_point][np.abs(Diffs_window[body_point]) 
                    < np.abs(treshold * (Diffs_max[body_point] - Diffs_min[body_point]) + Diffs_min[body_point]) ] = 0

    return Diffs_window, Diffs_max

def displacement_sum(Diffs):
    """Sum of displacement"""
    return np.sum(np.abs(Diffs), axis=1)

def displacement_var(Diffs):
    """Variance of displacement"""
    return np.var(Diffs, axis=1)

def scalars(Data):
    """
    Returns the average and variance position of the gesticulation in relation to the trunk
    Works for the first 4 arm bodypoints
    """
    Scals = np.empty((4, np.shape(Data)[1]))
    for row in range(4):
        for i, point in enumerate(Data[row]):
            Scals[row][i] = np.dot([point.x, point.y], [0, -1])
    return np.average(Scals, axis=1), np.var(Scals, axis=1)

def plot_data(data, title):
    """Helper for plotting"""
    plt.plot(data[0], 'orange', linewidth=1.4)
    plt.plot(data[1], 'orangered', linewidth=1.4)
    plt.plot(data[2], 'olive', linewidth=1.4)
    plt.plot(data[3], 'darkgreen', linewidth=1.4)
    plt.xlabel("frame")
    plt.ylabel("gesticulation")
    plt.legend(('right elbow', 'right hand', 'left elbow', 'left hand'))
    plt.title(title)
    plt.axis([0, np.shape(data)[1] - 1, -.3, .3])
    plt.savefig("Signed_avg_disp_"+ title +".png")
    plt.show()
