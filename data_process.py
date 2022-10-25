import numpy as np
import scipy 
from constants import NPOINTS

def resolve_zeros(Data):
    """
    Imputation of missing data with neighbouring data
    """
    lastframe_idx = np.shape(Data)[1] - 1
    data_length = np.shape(Data)[1]

    for body_point in range(NPOINTS):
        # Resolving zeros at the beginning of table
        if Data[body_point][0].x == 0 and Data[body_point][0].y == 0:
            end_idx = 1
            while Data[body_point][end_idx].x == 0 and Data[body_point][end_idx].y == 0: # Searching for the first not 0 coordinate beginning from the start
                if end_idx == lastframe_idx:
                    break
                end_idx += 1
            if end_idx == lastframe_idx:
                pass
            elif Data[body_point][end_idx + 1].x == 0: # If first not 0 point hasn't got any neighbours, than it is copied 
                for p in range(1, end_idx + 1):
                    Data[body_point][end_idx - p].x = Data[body_point][end_idx].x
                    Data[body_point][end_idx - p].y = Data[body_point][end_idx].y
            else: # If first not 0 point is subsequent by not 0 point, than the preceeding point are calculated by interpolation
                kulonbseg_x = Data[body_point][end_idx + 1].x - Data[body_point][end_idx].x
                kulonbseg_y = Data[body_point][end_idx + 1].y - Data[body_point][end_idx].y
                for p in range(1, end_idx + 1):
                    Data[body_point][end_idx - p].x = Data[body_point][end_idx].x - p * kulonbseg_x
                    Data[body_point][end_idx - p].y = Data[body_point][end_idx].y - p * kulonbseg_y

        # Resolving zeros at the end of table
        if Data[body_point][lastframe_idx].x == 0 and Data[body_point][lastframe_idx].y == 0:
            start_idx = lastframe_idx - 1
            while Data[body_point][start_idx].x == 0 and Data[body_point][start_idx].y == 0: # Searching for the first not 0 coordinate beginning from the end
                if start_idx == 0:
                    break
                start_idx -= 1
            if start_idx == 0:
                pass
            elif Data[body_point][start_idx - 1].x == 0: # Copying 
                for p in range(1, data_length - start_idx):
                    Data[body_point][start_idx + p].x = Data[body_point][start_idx].x
                    Data[body_point][start_idx + p].y = Data[body_point][start_idx].y
            else: # Interpolation
                diff_x = Data[body_point][start_idx].x - Data[body_point][start_idx - 1].x
                diff_y = Data[body_point][start_idx].y - Data[body_point][start_idx - 1].y
                for p in range(1, data_length - start_idx):
                    Data[body_point][start_idx + p].x = Data[body_point][start_idx].x + p * diff_x
                    Data[body_point][start_idx + p].y = Data[body_point][start_idx].y + p * diff_y

    start_idx = 0
    end_idx = 0
    for body_point in range(NPOINTS):
        # Resolving zeros in the middle of the table
        for j in range(1, data_length):
            if Data[body_point][j - 1].x != 0 and Data[body_point][j].x == 0: start_idx = j
            if Data[body_point][j - 1].x == 0 and Data[body_point][j].x != 0: end_idx = j - 1
            if start_idx > 0 and end_idx > 0:
                dist = end_idx - start_idx + 2
                vec_x = (Data[body_point][end_idx   + 1].x - Data[body_point][start_idx - 1].x) / dist
                vec_y = (Data[body_point][end_idx   + 1].y - Data[body_point][start_idx - 1].y) / dist
                for k in range(1, dist + 1):
                    Data[body_point][start_idx - 1 + k].x = Data[body_point][start_idx - 1].x + k * vec_x
                    Data[body_point][start_idx - 1 + k].y = Data[body_point][start_idx - 1].y + k * vec_y
                start_idx = 0
                end_idx = 0 
    return Data

def savgol_filter(Data, window_length, polyorder):
    """
    Savitzky-Golay algorithm for filtering data in time for every body point
    """
    for row in range(NPOINTS):
        new_row_x = [point.x for point in Data[row]]
        new_row_y = [point.y for point in Data[row]]
        new_row_x = scipy.signal.savgol_filter(new_row_x, window_length, polyorder)
        new_row_y = scipy.signal.savgol_filter(new_row_y, window_length, polyorder)
        for body_point, frame in enumerate(Data[row]):
            frame.x = new_row_x[body_point]
            frame.y = new_row_y[body_point]
    return Data
    