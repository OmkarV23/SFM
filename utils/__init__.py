import numpy as np
import cupy as cp

def camera_mat(file):
    mat_file = open(file,'r')
    mat = np.array([float(j) for i in mat_file.readlines() for j in i[:-1].split(' ')]).reshape(3,3)
    return mat

def projection_mat(K,Rot,trans):
    proj_mat = K @ Rot @ np.concatenate([cp.eye(3), trans], axis=1)
    return proj_mat

def Essential_mat(K,F):
    E = K.T @ F @ K
    u,_,vt = cp.linalg.svd(E)
    E = u @ cp.diag([1,1,0]) @ vt
    return E