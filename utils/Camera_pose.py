import cupy as cp

def pose(E):
    
    cheirality_freq = []
    W = cp.array([[0,-1,0],[1,0,0],[0,0,1]])
    U,D,Vt = cp.linalg.svd(E)

    camera_poses = {'1':{'C':U[:,2], 'R':U @ W @ Vt},
                    '2':{'C':-U[:,2], 'R':U @ W @ Vt},
                    '3':{'C':U[:,2], 'R':U @ W.T @ Vt},
                    '4':{'C':-U[:,2], 'R':U @ W.T @ Vt}}
    for i in camera_poses.keys():
        if cp.linalg.det(camera_poses[i]['R']).astype(cp.int8)==-1:
            camera_poses[i]['R'] *= -1
            camera_poses[i]['C'] *= -1
        
    return camera_poses

    
    
    


    
# print(pose(cp.array([[0.75599978,-0.56102436,-0.18035228],
# [0.44184109,0.81747558,-0.19880999],
#  [ 0.38940319,0.10448964,-0.12498331]])))