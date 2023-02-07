import numpy as np

def camera_mat(file):
    mat_file = open(file,'r')
    mat = np.array([float(j) for i in mat_file.readlines() for j in i[:-1].split(' ')]).reshape(3,3)
    return mat