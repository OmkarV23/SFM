import sys, random
import cupy as cp
import numpy as np
import cv2
from EstimateFundamentalMatrix import F

N = 9
threshold = 0.005
iteration_limit = 2000
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
best_inliers = {'RGB values':[],'u1':[],'u2':[]}
for i in range(iteration_limit):
    idxs = random.sample(range(0, len(matching_features['RGB values'])), N)
    rgb = [matching_features['RGB values'][i] for i in idxs]
    u1 = [matching_features['u1'][i] for i in idxs]
    u2 = [matching_features['u2'][i] for i in idxs]
    inliers,Fundamental_mat = F(rgb,u1,u2,threshold)
    if inliers!=None:
        best_inliers['RGB values'].append(inliers['RGB values'])
        best_inliers['u1'].append(inliers['u1'][0])
        best_inliers['u2'].append(inliers['u2'][0])
        inlier_count = len(inliers)
        if inlier_count > max_inlier_count:
            max_inlier_count = inlier_count
            best_Fundamental_mat = Fundamental_mat

img_1 = cv2.imread('/workspace/omkar_projects/WPI_CV/SFM/P3Data/1.png')
img_2 = cv2.imread('/workspace/omkar_projects/WPI_CV/SFM/P3Data/2.png')
image_pts_1 = cp.array(best_inliers['u1'])[:,:-1].get()
image_pts_2 = cp.array(best_inliers['u2'])[:,:-1].get()
for i,j in zip(image_pts_1,image_pts_2):
    pt_1 = tuple(i)
    pt_2 = tuple(j)
    img_1 = cv2.circle(img_1,(round(pt_1[0]),round(pt_1[1])), 2, (0,0,255),thickness=-1)
    img_2 = cv2.circle(img_2,(round(pt_2[0]),round(pt_2[1])), 2, (0,0,255),thickness=-1)
cv2.imwrite('kncvjdkjvcjn.jpg', img_1)
cv2.imwrite('kncvjdkjvcjm.jpg', img_2)
