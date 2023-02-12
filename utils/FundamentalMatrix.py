import cupy as cp
import sys
import cv2


def scale_mat(scales):
    """scales: scaling fractor for the image 1 and image 2 after centre correction"""
    """Shape of scales: (1,2), [scaling factor for image 1, scaling factor for image 2]"""
    """Returns two scale matrices for each image"""
    
    scale_matrices = []
    for i in range(len(scales)):
        scale_matrices.append(cp.array([[scales[i].get(),0,0],
                                        [0,scales[i].get(),0],
                                        [0,0,1]]))
    return scale_matrices[0], scale_matrices[1]


def translation_mat(mean):
    """mean: mean of x1, y1, x2, y2 for all points in the 2 images"""
    """Shape of mean: (1,6), [mean_X1, mean_y1, 1, mean_x2, mean_y2, 1]"""
    """Returns the translation matrices for each images"""

    trans1 = cp.array([[1,0,-mean[0].get()],[0,1,-mean[1].get()],[0,0,1]])
    trans2 = cp.array([[1,0,-mean[3].get()],[0,1,-mean[4].get()],[0,0,1]])

    return trans1, trans2


def normalize(points):
    pts_mean = cp.mean(points, axis=0)
    mean_correction = points-pts_mean
    scale = (len(points) / cp.mean((mean_correction[:,0])**2 + (mean_correction[:,1])**2))**(0.5)
    scale_mat = cp.array([[scale.get(),0,0],
                        [0,scale.get(),0],
                        [0,0,1]])
    translation_mat = cp.array([[1,0,-pts_mean[0].get()],[0,1,-pts_mean[1].get()],[0,0,1]])
    transformation_mat = cp.matmul(scale_mat, translation_mat)
    return (transformation_mat @ points.T).T, transformation_mat

def F(rgb_lst,u1_lst,u2_lst,threshold=0.003):

    u1_normalized, transformation_mat1 = normalize(cp.array(u1_lst))
    u2_normalized, transformation_mat2 = normalize(cp.array(u2_lst))

    mat = []
    for u1,u2 in zip(u1_normalized,u2_normalized):
        eq = cp.concatenate([cp.dot(u1,u2[0]), cp.dot(u1,u2[1]), cp.dot(u1,u2[2])],axis=0)
        mat.append(eq)
    mat = cp.array(mat)
    _,s,vT = cp.linalg.svd(mat, full_matrices=True)

    # initial F
    F_init = vT.T[:,-1].reshape(3,3)

    # Denormalize F
    F_denorm = transformation_mat2.T @ F_init @ transformation_mat1

    # SVD of denormalized fundamental matrix
    u_f,s_f,vt_f = cp.linalg.svd(F_denorm)
    
    # SVD cleanup
    s_f = cp.diag(s_f)
    s_f[2,2] = 0
    
    # reconstruct F
    F = u_f @ s_f @ vt_f
    F = F/F[2,2]

    inliers = {'RGB values':[], 'u1':[], 'u2':[]}
    for u1,u2 in zip(u1_lst,u2_lst):
        if abs(cp.array(u1)[None,:] @ F @ cp.array(u2)[:,None]).squeeze(1)[0] < threshold:
            inliers['RGB values'].append(rgb_lst[u1_lst.index(u1)])
            inliers['u1'].append(u1)
            inliers['u2'].append(u2)
    if len(inliers['RGB values'])==0:
        return None, F
    else:
        return inliers, F