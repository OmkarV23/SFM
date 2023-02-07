import cupy as cp
from utils import camera_mat, projection_mat, Essential_mat
from utils.RANSAC import RANSAC
from utils.Camera_pose import pose
from utils.linear_triangulation import LinearTriangulation

threshold = 0.009
iteration_limit = 2000
N = 10

K = cp.asarray(camera_mat('/workspace/omkar_projects/WPI_CV/SFM/P3Data/calibration.txt'))
R = RANSAC(threshold,iteration_limit,N)
inliers, F = R.inliers()
E = Essential_mat(K,F)
camera_poses = pose(E)
origin_trans = cp.zeros(3)
original_rot = cp.eye(3)
for p in camera_poses.keys():
    world_pts = LinearTriangulation(K, origin_trans, original_rot, camera_poses[p]['C'], camera_poses[p]['R'], inliers)
    print(world_pts)

