import os
import dlib
import csv
import numpy as np
import pandas as pd
import logging
import cv2

# Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()
# Dlib 人脸 68个特征点检测器 预测器
predictor = dlib.shape_predictor('./model/shape_predictor_68_face_landmarks.dat')


def face_reco(img_path):
    img = cv2.imread(img_path)

    face = detector(img, 0)  # face中的数据是：rectangles[[(198, 130) (508, 439)]]
    print(face)
    print(type(face))    # 类型是detector定义的正方形变量
    # 在脸部画出正方形
    for h in face:
        print(h)
        cv2.rectangle(img, (h.left(), h.top()), (h.right(), h.bottom()), (0, 255, 255), 2)

    cv2.imshow("jpg1", img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    cv2.imshow("gray", gray_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def write_csv():
    img = cv2.imread("./images/person1.jpg")
    print(type(img))
    arr = np.array(img)
    print(len(arr[0]))
    with open("./data/jpg.csv", "w", newline='') as file:
        write = csv.writer(file)
        write.writerows(arr)
        print(file)

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    cv2.imwrite("./images/gray_img.jpg", gray_img)  # 保存图片
    read_gray = cv2.imread("./images/gray_img.jpg")

    arr_gray = np.array(read_gray)

    with open("./data/jpg_gray.csv", "w", newline='') as file1:
        write1 = csv.writer(file1)
        write1.writerows(arr_gray)


if __name__ == "__main__":
    face_reco("./images/person1.jpg")
    write_csv()

