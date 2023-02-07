import cupy as cp
from utils import projection_mat

def skew(pts):
    return cp.array([[0, -pts[2], pts[1]], [pts[2], 0, -pts[0]], [-pts[1], pts[0], 0]])

def LinearTriangulation(K, C1, R1, C2, R2, inliers):
    world_pts = []
    proj_mat1 = projection_mat(K,R1,C1[:,None])
    proj_mat2 = projection_mat(K,R2,C2[:,None])
    u1 = inliers['u1']
    u2 = inliers['u2']

    for i,j in zip(u1,u2):
        mat = cp.vstack((skew(i) @ proj_mat1, skew(j) @ proj_mat2))
        u,s,vt = cp.linalg.svd(mat)
        X = vt[-1] / vt[-1,-1]
        X = cp.reshape(X, (len(X), -1)).transpose()
        world_pts.append(X)
    world_pts = cp.array(world_pts)
    return world_pts