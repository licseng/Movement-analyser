import numpy as np
from sympy.solvers import solve
from sympy import Symbol

class Point2d(object):
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y

class Point3d(object):
    def __init__(self, x_coord, y_coord, z_coord):
        self.x = x_coord
        self.y = y_coord
        self.z = z_coord

def trunk2d(Data):
    """
    Returns the transformation matrix for changing from frame coordinate system to body coordinate system in 2d.
    Additionally returns the two trunk points for testing purposes
    """
    x_neck_ref      = Data['people'][0]['pose_keypoints_2d'][3]  
    y_neck_ref      = Data['people'][0]['pose_keypoints_2d'][4]  
    x_hip_ref       = Data['people'][0]['pose_keypoints_2d'][24]
    y_hip_ref       = Data['people'][0]['pose_keypoints_2d'][25]
    x_lshoulder_ref = Data['people'][0]['pose_keypoints_2d'][15]
    y_lshoulder_ref = Data['people'][0]['pose_keypoints_2d'][16]

    ref_bottom = [x_hip_ref, y_hip_ref]
    ref_origo = [x_neck_ref, y_neck_ref]

    r2 = np.array([x_hip_ref - x_neck_ref, y_hip_ref - y_neck_ref])

    r1_lshoulder = np.array([x_lshoulder_ref - x_neck_ref, y_lshoulder_ref - y_neck_ref])
    r2_perpend = np.array([(y_hip_ref - y_neck_ref) * -1, (x_hip_ref - x_neck_ref)])
    r2_perpend_norm = np.linalg.norm(r2_perpend)
    r2_perpend_unit = r2_perpend / r2_perpend_norm
    r1_length = np.dot(r1_lshoulder, r2_perpend_unit)
    r1 = r1_length * r2_perpend_unit

    B = np.array([r1, -r2])

    # Checks if transformation matrix exsits, if not returns matrix of zeros
    if np.linalg.det(B) == 0:
        return np.zeros([2, 2]), ref_origo, ref_bottom
    else:
        return np.linalg.inv(np.transpose(B)), ref_origo, ref_bottom


def trunk3d(Data, idx):
    """
    Returns the transformation matrix for changing from frame coordinate system to body coordinate system in 3d.
    """
    x_neck_ref      = Data[str(idx)][str(13)]["translate"][0]        
    y_neck_ref      = Data[str(idx)][str(13)]["translate"][1]        
    z_neck_ref      = Data[str(idx)][str(13)]["translate"][2]        
    x_hip_ref       = Data[str(idx)][str(0)]["translate"][0]
    y_hip_ref       = Data[str(idx)][str(0)]["translate"][1]
    z_hip_ref       = Data[str(idx)][str(0)]["translate"][2]
    x_lshoulder_ref = Data[str(idx)][str(17)]["translate"][0]
    y_lshoulder_ref = Data[str(idx)][str(17)]["translate"][1]
    z_lshoulder_ref = Data[str(idx)][str(17)]["translate"][2]

    ref_origo =  [x_neck_ref, y_neck_ref, z_neck_ref]

    r2 = np.array([x_hip_ref - x_neck_ref, y_hip_ref - y_neck_ref, z_hip_ref - z_neck_ref], dtype=np.float32)
    
    # We have an equation of the plane perpendicular to the trunk at the height of the neck,
    # and three equations for the perpendicular that passes through the shoulder.
    # The projection of the shoulder to the trunk can be calculated with these equations.
    x = Symbol('x') 
    x_r1_lshoulder = solve(r2[0]*(x-x_neck_ref) + r2[1]*((y_lshoulder_ref + r2[1]*(x-x_lshoulder_ref)/r2[0])-y_neck_ref) + r2[2]*((z_lshoulder_ref + r2[2]*(x-x_lshoulder_ref)/r2[0])-z_neck_ref),x )[0]
    y_r1_lshoulder = (y_lshoulder_ref + r2[1]*(x_r1_lshoulder-x_lshoulder_ref)/r2[0])
    z_r1_lshoulder = (z_lshoulder_ref + r2[2]*(x_r1_lshoulder-x_lshoulder_ref)/r2[0])
    r1 = np.array([x_r1_lshoulder - x_neck_ref, y_r1_lshoulder - y_neck_ref, z_r1_lshoulder - z_neck_ref],dtype=np.float32)

    r3 = np.cross(r1, r2)
    r1_len = np.linalg.norm(r1)
    len = solve(np.square(r3[0]*x) + np.square(r3[1]*x) + np.square(r3[2]*x) - np.square(r1_len), x)
    r3 = r3*len[1]

    B = np.array([r1, -r3, -r2], dtype='float')

    if np.linalg.det(B) == 0:
        return np.zeros([3, 3]), ref_origo
    else:
        return np.linalg.inv(np.transpose(B)), ref_origo

def transform(B_transf, point, o, mode = "2d"):
    """
    Gets the current frame's transformation matrix to transform point to new coordinate system.
    Returns the new coordinates.
    Can be used in "2d" and "3d" mode.
    """
    if (np.all(point) == 0 or np.all(B_transf) == 0): 
        if mode == "2d":
            return Point2d(0, 0)  
        elif mode == "3d":
            return Point3d(0, 0, 0)
    point = np.subtract(point, o)
    point = B_transf.dot(point)
    if mode == "2d":
        return Point2d(point[0], point[1])
    elif mode == "3d":
            return Point3d(point[0], point[1], point[2])
