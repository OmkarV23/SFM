from utils import camera_mat
from utils.RANSAC import RANSAC
from utils.EssentialMatrix import Essential_mat

threshold = 0.009
iteration_limit = 2000
N = 12

K = camera_mat('/workspace/omkar_projects/WPI_CV/SFM/P3Data/calibration.txt')
R = RANSAC(threshold,iteration_limit,N)
inliers, F = R.inliers()
E = Essential_mat(K,F)
print(E)


