import cupy as cp
import sys


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


def F(rgb_lst,u1_lst,u2_lst,threshold=0.003):
    
    # Concenate the points from image 1 and image 2 for easy operations and leverare cupy/numpy functions.
    # Concenate across the rows.
    points_1_2 = cp.concatenate([cp.array(u1_lst),cp.array(u2_lst)], axis=1)
    
    # Take the mean across the columns to get a 1X6 vector [x1_mean, y1_mean, 1, x2_mean, y2_mean, 1]
    mean_1_2 = cp.mean(points_1_2, axis=0   )[None,:]

    # Point correction with (0,0) at the centroid of the image. Get x1_dash,y1_dash,x_2dash,y_2dash. 
    corrected_pts = points_1_2 - mean_1_2

    # calculate scales using L2 norm
    scale_1 = len(u1_lst) / cp.sum(((corrected_pts[0])**2 + (corrected_pts[1])**2)**(0.5))
    scale_2 = len(u2_lst) / cp.sum(((corrected_pts[3])**2 + (corrected_pts[4])**2)**(0.5))

    # refer 'scale_mat' and 'translation_mat' for the explanation
    scale1, scale2 = scale_mat([scale_1, scale_2])

    translation1, translation2 = translation_mat(mean_1_2.squeeze(0))

    # get transformation matrix using scale and translation
    transformation1, transformation2 = cp.matmul(scale1,translation1), cp.matmul(scale2,translation2)

    # multiply with original points to get the transformed points
    u1_normalized = (transformation1 @ points_1_2.T[:3]).T
    u2_normalized = (transformation2 @ points_1_2.T[3:]).T

    mat = []
    for u1,u2 in zip(u1_normalized,u2_normalized):
        u1 = cp.array(u1)
        u2 = cp.array(u2)
        eq = cp.concatenate([cp.dot(u1,u2[0]), cp.dot(u1,u2[1]), cp.dot(u1,u2[2])])
        mat.append(eq)
    mat = cp.array(mat)
    _,s,vT = cp.linalg.svd(mat, full_matrices=True)

    # initial F
    F_init = vT.T[:,-1].reshape(3,3)

    # Denormalize F
    F_denorm = transformation2.T @ F_init @ transformation1

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