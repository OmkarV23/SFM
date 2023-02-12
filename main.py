import cupy as cp
from utils import camera_mat, projection_mat, Essential_mat
from utils.RANSAC import RANSAC
from utils.Camera_pose import pose, cheirality
from utils.triangulation import Triangulation
import matplotlib.pyplot as plt
import sys
import cv2

threshold = 0.009
iteration_limit = 2000
N = 15

K = cp.asarray(camera_mat('/workspace/omkar_projects/WPI_CV/SFM/P3Data/calibration.txt'))
R = RANSAC(threshold,iteration_limit,N)
inliers, F = R.inliers()
E = Essential_mat(K,F)
camera_poses = pose(E)

origin_trans = cp.zeros(3)
original_rot = cp.eye(3)

cheirality_counter = 0
world_pts = None
best_pose = None

T  = Triangulation(K, origin_trans, original_rot, inliers)

## Do linear Triangulation and Cheirality check

for p in camera_poses.keys():    
    X_init = T.LinearTriangulation(camera_poses[p]['C'], camera_poses[p]['R'])
    number_of_positives = cheirality(camera_poses[p]['C'], camera_poses[p]['R'], X_init)
    if number_of_positives > cheirality_counter:
        cheirality_counter = number_of_positives
        world_pts = X_init.squeeze(1)#[:,:-1]
        best_pose = camera_poses[p]

## From the initial estimate of best pose, do non linear triangulation

X = T.NonLinearTriangulation(best_pose['C'], best_pose['R'], world_pts)
X_refined = X / X[:,3].reshape(-1,1)
print(X_refined)

# #######################################################
proj1 = K @ original_rot @ cp.concatenate([cp.eye(3), origin_trans[:,None]], axis=1)
proj2 = K @ best_pose['R'] @ cp.concatenate([cp.eye(3), best_pose['C'][:,None]], axis=1)

u1 = cp.array(inliers['u1'])[:,:-1]
u2 = cp.array(inliers['u2'])[:,:-1]

# u1_proj = cp.divide(proj1[0][None,:] @ world_pts.T, proj1[2][None,:] @ world_pts.T)
# v1_proj = cp.divide(proj1[1][None,:] @ world_pts.T, proj1[2][None,:] @ world_pts.T)

u2_proj = cp.divide(proj2[0][None,:] @ world_pts.T, proj2[2][None,:] @ world_pts.T)
v2_proj = cp.divide(proj2[1][None,:] @ world_pts.T, proj2[2][None,:] @ world_pts.T)

u2_proj_ref = cp.divide(proj2[0][None,:] @ X_refined.T, proj2[2][None,:] @ X_refined.T)
v2_proj_ref = cp.divide(proj2[1][None,:] @ X_refined.T, proj2[2][None,:] @ X_refined.T)

img = cv2.imread('/workspace/omkar_projects/WPI_CV/SFM/P3Data/2.png')
for i,j,k,l,m in zip(u2_proj[0], v2_proj[0], u2, u2_proj_ref[0], v2_proj_ref[0]):
    if i>0 and j>0:
        img = cv2.circle(img,(round(i.get().item()),round(j.get().item())),1,(0,0,255),-1)
    if l>0 and m>0:
        img = cv2.circle(img,(round(l.get().item()),round(m.get().item())),1,(255,0,0),-1)
    img = cv2.circle(img,(round(k[0].get().item()),round(k[1].get().item())),1,(0,255,0),-1)
# for x in u2:
#     img = cv2.circle(img,(round(x[0].get().item()),round(x[1].get().item())),1,(0,255,0),-1)
cv2.imwrite('dnvkjfnv.jpg', img)

# # print('##################U1#######################')
# # print(u1[:,0][100])
# # print('##################U1_reproj#################')
# # print(u1_proj[0][100])


# #######################################################



# print(world_pts[:,0].shape)

# fig1= plt.figure(figsize= (5,5))
# ax = plt.axes(projection="3d")
# ax.scatter3D(world_pts[:,0].get(),world_pts[:,1].get(),world_pts[:,2].get(),color="green")
# plt.savefig('3D.png')

# fig = plt.figure(figsize = (10,10))
# plt.xlim(-4,6)
# plt.ylim(-2,12)
# plt.scatter(world_pts[:,0].get(),world_pts[:,1].get(),marker='.',linewidths=0.5, color = 'blue')
# plt.savefig('2D.png')