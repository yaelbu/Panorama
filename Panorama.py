from datetime import datetime
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

#part1
start_time = datetime.now()
print("start time : ")
print(start_time)
path_left_image=sys.argv[1]
print("path_left_image    :    ")
print(path_left_image)
path_right_image=sys.argv[2]
print("path_right_image  :  ")
print(path_right_image)
path_output_image=sys.argv[3]
print("path_output_image   :   ")

print(path_output_image)

img_left = cv2.imread(path_left_image, cv2.IMREAD_COLOR)
img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
img_right = cv2.imread(path_right_image, cv2.IMREAD_COLOR)
img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
cv2.imshow("kokoleft",img_left)
cv2.imshow("kikiright",img_right)
img_left_gray=cv2.resize(img_left_gray,(int(img_left_gray.shape[0]*0.12),int(img_left_gray.shape[1]*0.12)))
img_right_gray=cv2.resize(img_right_gray,(int(img_right_gray.shape[0]*0.12),int(img_right_gray.shape[1]*0.12)))



'''
print(img_left_gray.shape[0])
print("len left gray\n")
print(img_right_gray.shape[0])
print("len right gray\n")
print(img_left_gray.shape[1])
print("width left gray\n")
print(img_right_gray.shape[1])
print("witdh right gray\n")

print("original:\n")
print(img_left.shape[0])
print("len left gray\n")
print(img_right.shape[0])
print("len right gray\n")
print(img_left.shape[1])
print("width left gray\n")
print(img_right.shape[1])
print("witdh right gray\n")
'''

if img_left_gray.shape[0]>=img_right_gray.shape[0]:


    img_right_gray = cv2.resize(img_right_gray, (img_right_gray.shape[1],img_left_gray.shape[0]))
    img_left_gray=cv2.resize(img_left_gray,(img_left_gray.shape[1],img_left_gray.shape[0]))
    img_right = cv2.resize(img_right, (img_right_gray.shape[1], img_left_gray.shape[0]))
    img_left = cv2.resize(img_left, (img_left_gray.shape[1], img_left_gray.shape[0]))

else:

    img_left_gray = cv2.resize(img_left_gray, (img_left_gray.shape[1], img_right_gray.shape[0]))
    img_right_gray = cv2.resize(img_right_gray, (img_right_gray.shape[1], img_right_gray.shape[0]))
    img_left = cv2.resize(img_left, (img_left_gray.shape[1], img_right_gray.shape[0]))
    img_right = cv2.resize(img_right, (img_right_gray.shape[1], img_right_gray.shape[0]))

'''
print(img_left_gray.shape[0])
print("len left gray\n")
print(img_right_gray.shape[0])
print("len right gray\n")
print(img_left_gray.shape[1])
print("width left gray\n")
print(img_right_gray.shape[1])
print("witdh right gray\n")

print("original:\n")
print(img_left.shape[0])
print("len left\n")
print(img_right.shape[0])
print("len right\n")
print(img_left.shape[1])
print("width left\n")
print(img_right.shape[1])
print("witdh right\n")
'''

sift_img_left = cv2.xfeatures2d.SIFT_create()
sift_img_right = cv2.xfeatures2d.SIFT_create()

keypoints_left, descriptors_left = sift_img_left.detectAndCompute(img_left_gray,None)
keypoints_right, descriptors_right = sift_img_right.detectAndCompute(img_right_gray,None)


#part 2

matcher = cv2.BFMatcher()
raw_matches = matcher.knnMatch(descriptors_left, descriptors_right, k=2)


#part 3

ratio=0.85
matches_list=[]
matches_without_list=[]

for m1,m2 in raw_matches:
    if m1.distance < ratio*m2.distance:
        matches_list.append([m1])
        matches_without_list.append(m1)



imMatches = cv2.drawMatchesKnn(img_left_gray, keypoints_left, img_right_gray, keypoints_right,matches_list, None,flags=2)
cv2.imshow('imMatches', imMatches)
cv2.waitKey(0)
cv2.destroyAllWindows()





#part 4

keypoints_left1 = np.float32([keypoints_left[m.queryIdx].pt for m in matches_without_list]).reshape(-1,1,2)
keypoints_right1 = np.float32([keypoints_right[m.trainIdx].pt for m in matches_without_list]).reshape(-1,1,2)
H, status = cv2.findHomography(keypoints_right1, keypoints_left1, cv2.RANSAC, 5.0)

'''
print(img_right_gray.shape)
print(img_right.shape)
print(img_left_gray.shape)
print(img_left.shape)
'''


#part5
width_panorama=img_left_gray.shape[1] + img_right_gray.shape[1]
height_panorama=img_left_gray.shape[0]
result = cv2.warpPerspective(img_right, H, (width_panorama,height_panorama),flags=cv2.INTER_LINEAR)
result[0:img_left.shape[0], 0:img_left.shape[1]] = img_left

#result = cv2.warpPerspective(img_left,H,(470,1000))
#cv2.imwrite(path_output_image,result)
#plt.imshow(result)
#plt.show()
#cv2.waitKey(0)
#cv2.destroyAllWindows()




gray = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
_,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
x,y,w,h = cv2.boundingRect(cnt)
crop = result[y:y+h,x:x+w]
cv2.imwrite(path_output_image,crop)
plt.imshow(crop)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()




end_time=datetime.now()
print("end time : ")
print(end_time)
print("Duration : {}".format(end_time-start_time))
