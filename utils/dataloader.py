import sys, random
import cupy as cp
import cv2

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
        matching_features['u2'].append((float(l[l[1:].index(str(number_of_images))+2]),
                                        float(l[l[1:].index(str(number_of_images))+3]),1))
        

img = cv2.imread('/workspace/omkar_projects/WPI_CV/SFM/P3Data/1.png')
for i in matching_features['u1']:
    x = round(i[0])
    y = round(i[1])
    img = cv2.circle(img, (x,y), 2, (0,255,0), -1)
cv2.imwrite('IMG2.jpg', img)