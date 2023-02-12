import cupy as cp
import sys

def pose(E):
    
    W = cp.array([[0,-1,0],[1,0,0],[0,0,1]])
    U,D,Vt = cp.linalg.svd(E)

    camera_poses = {'1':{'C':U[:,2], 'R':(U @ (W @ Vt))},
                    '2':{'C':-U[:,2], 'R':(U @ (W @ Vt))},
                    '3':{'C':U[:,2], 'R':(U @ (W.T @ Vt))},
                    '4':{'C':-U[:,2], 'R':(U @ (W.T @ Vt))}}
    for i in camera_poses.keys():
        if cp.linalg.det(camera_poses[i]['R']).astype(cp.int8)==-1:
            camera_poses[i]['R'] *= -1
            camera_poses[i]['C'] *= -1
        
    return camera_poses

def cheirality(C,R,X):
    counter = 0
    for world_point in X:
        world_point = world_point[0] / world_point[0][-1]
        if (R[:,2][None,:] @ (world_point[:-1][:,None] - C[:,None])).squeeze(1).item() > 0\
            and world_point[2]>0:
            counter+=1
    return counter   

