import numpy as np
import cv2, dlib, sys, imutils, math

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/face_landmark.dat')
'''
# 0~26: 얼굴 윤곽
# 27~30: 콧대
# 31~35: 코아래
# 36~41: 왼눈
# 42~47: 오른눈
# 48~ : 입
'''


class get_face():
    def __init__(self, img):
        # self.img = cv2.imread(img)
        self.img = img
        self.face = detector(self.img)[0]
        self.dlib_shape = predictor(self.img, self.face)
        self.shape_2d = np.array([[p.x, p.y] for p in self.dlib_shape.parts()])

        # 얼굴 중심값
        self.center_x, self.center_y = np.mean(self.shape_2d, axis=0).astype(np.int)

        # 얼굴 각도
        l_eye_x, l_eye_y = np.mean(self.shape_2d[36:42], axis=0).astype(np.int)
        r_eye_x, r_eye_y = np.mean(self.shape_2d[42:48], axis=0).astype(np.int)  # 눈 각

        n_top_x, n_top_y = self.shape_2d[27]
        n_bottom_x, n_bottom_y = self.shape_2d[30]  # 코 수직각

        l_n_x, l_n_y = self.shape_2d[31]
        r_n_x, r_n_y = self.shape_2d[35]  # 코 수평각

        degree1 = math.atan((r_eye_y - l_eye_y) / (r_eye_x - l_eye_x))
        degree2 = math.atan((n_top_x - n_bottom_x) / (n_bottom_y - n_top_y))
        degree3 = math.atan((r_n_y - l_n_y) / (r_n_x - l_n_x))
        degree = np.mean([degree1, degree2, degree3])
        self.degree = np.rad2deg(degree)

        # 얼굴 크기(눈 거리)
        self.eye_dis = ((r_eye_y - l_eye_y) ** 2 + (r_eye_x - l_eye_x) ** 2) ** 0.5
        self.top_left = np.min(self.shape_2d, axis=0)
        self.bottom_right = np.max(self.shape_2d, axis=0)
        self.len_x = self.bottom_right[0]-self.top_left[0]
        self.len_y = self.bottom_right[1] - self.top_left[1]


def get_info(img):
    face = detector(img)[0]
    dlib_shape = predictor(img, face)
    shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])
    center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)
    return center_x, center_y


if __name__ == '__main__':
    img = cv2.imread('images/face.jpg')
    img = cv2.resize(img, (int(img.shape[1] * 0.3), int(img.shape[0] * 0.3)))
    gf = get_face(img)

    img = imutils.rotate(img, gf.degree, center=(gf.center_x, gf.center_y))

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
