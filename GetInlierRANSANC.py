import sys, random
import cupy as cp
import cv2
from EstimateFundamentalMatrix import F

N = 100
threshold = 0.01
iteration_limit = 100
max_inlier_count = 0


best_Fundamental_mat = None
best_inliers = None
number_of_images = 2

matching_file = '/workspace/omkar_projects/WPI_CV/SFM/P3Data/matching1.txt'

matching_features = {'RGB values':[],'u1':[],'u2':[]}

file = open(matching_file, 'r')
for idx,line in enumerate(file.readlines()[1:]):
    l = line[:-2].split(' ')
    img_id = [int(l[(i+1)*3]) for i in range(int(l[0]))][1:]
    if number_of_images in img_id:
        matching_features['RGB values'].append((int(l[1]),int(l[2]),int(l[3])))
        matching_features['u1'].append((float(l[4]),float(l[5]),1))
        matching_features['u2'].append((float(l[l.index(str(number_of_images))+1]),
                                        float(l[l.index(str(number_of_images))+2]),1))


for i in range(iteration_limit):
    idxs = random.sample(range(0, len(matching_features['RGB values'])), N)
    rgb = [matching_features['RGB values'][i] for i in idxs]
    u1 = [matching_features['u1'][i] for i in idxs]
    u2 = [matching_features['u2'][i] for i in idxs]
    inliers,Fundamental_mat = F(rgb,u1,u2,threshold)
    inlier_count = len(inliers)
    if inlier_count > max_inlier_count:
        max_inlier_count = inlier_count
        best_Fundamental_mat = Fundamental_mat
        best_inliers = inliers