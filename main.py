import numpy as np
import cv2, dlib, sys, imutils, math
import get_face
import glob

try:
    dir = sys.argv[1]
    r_name = sys.argv[2]
except:
    dir = 'images'
    r_name = 'result.png'

image_files = glob.glob(dir+'/*')
n = len(image_files)
images = []
gfs = []

#get imfo and rotate process
for image in image_files:
    img = cv2.imread(image)
    gf = get_face.get_face(img)
    gfs.append(gf)

    img = imutils.rotate(img, gf.degree, center=(gf.center_x, gf.center_y))
    images.append(img)

#get size
size_list = np.array([gf.len_x for gf in gfs])
min_x = np.min(size_list)
min_idx = np.argmin(size_list)

min_eye_dis = gfs[min_idx].eye_dis

#result size
result_x = (gfs[min_idx].center_x - gfs[min_idx].top_left[0])*1.8
result_y = (gfs[min_idx].center_y - gfs[min_idx].top_left[1])*1.8
result_x, result_y = int(result_x), int(result_y)

#resize, crop
result_images = []
for img, gf in zip(images, gfs):
    img = cv2.resize(img, dsize=None, fx=min_eye_dis/gf.eye_dis, fy=min_eye_dis/gf.eye_dis)
    center_x, center_y = get_face.get_info(img)
    img = img[center_y-result_y:center_y+result_y, center_x-result_x:center_x+result_x]

    result_images.append(img)


result = cv2.addWeighted(result_images[0], 1/n, result_images[1], 1/n, 0)
for img in result_images[2:]:
    result = cv2.addWeighted(result, 1, img, 1/n, 0)

cv2.imwrite('result/'+r_name, result)

