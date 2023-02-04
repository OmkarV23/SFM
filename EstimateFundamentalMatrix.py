import cupy as cp
import sys

def F(rgb_lst,u1_lst,u2_lst,threshold=0.003):
    mat = []
    for u1,u2 in zip(u1_lst,u2_lst):
        u1 = cp.array(u1)
        u2 = cp.array(u2)
        eq = cp.concatenate([cp.dot(u2,u1[0]), cp.dot(u2,u1[1]), cp.dot(u2,u1[2])])
        mat.append(eq)
    mat = cp.array(mat)
    _,_,vT = cp.linalg.svd(mat)
    F = vT.T[:,-1].reshape(3,3)
    inliers = {'RGB values':[], 'u1':[], 'u2':[]}
    for u1,u2 in zip(u1_lst,u2_lst):
        if abs(cp.array(u1)[None,:] @ F @ cp.array(u2)[:,None]).squeeze(1)[0] < threshold:
            inliers['RGB values'].append(rgb_lst[u1_lst.index(u1)])
            inliers['u1'].append(u1)
            inliers['u2'].append(u2)
    return inliers, F
    # if len(inliers) > max_inlier_count:
    #     max_inlier_count = len(inliers)
    # return inliers