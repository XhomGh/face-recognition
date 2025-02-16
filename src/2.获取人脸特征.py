# 其二：从人脸图像文件中提取人脸特征存入 "features_all.csv"

import os
import dlib
import csv
import numpy as np
import logging
import cv2

# 要读取人脸图像文件的路径
path_images_from_camera = "data/data_face/"
# Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()
# Dlib 人脸 68个特征点检测器 预测器
predictor = dlib.shape_predictor('./model/shape_predictor_68_face_landmarks.dat')
# Dlib 通过 Resnet 人脸识别模型，提取 128D 的特征矢量
face_reco_model = dlib.face_recognition_model_v1("./model/dlib_face_recognition_resnet_model_v1.dat")


# 返回单张图像的 128D 特征
# Input:    path_img           <class 'str'>
# Output:   face_descriptor    <class 'dlib.vector'>
def return_128d_features(path_img):
    img_rd = cv2.imread(path_img)  # img_rd就是将图片转为灰色
    faces = detector(img_rd, 1)

    logging.info("%-40s %-20s", "检测到人脸的图像：", path_img)

    # 因为有可能截下来的人脸再去检测，检测不出来人脸了, 所以要确保是 检测到人脸的人脸图像拿去算特征
    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
    else:
        face_descriptor = 0
        logging.warning("no face")
    return face_descriptor


# 返回 personX 的 128D 特征均值
# Input:    path_face_personX        <class 'str'>
# Output:   features_mean_personX    <class 'numpy.ndarray'>
def return_features_mean_personX(path_face_personX):
    features_list_personX = []
    photos_list = os.listdir(path_face_personX)
    if photos_list:
        for i in range(len(photos_list)):
            # 调用 return_128d_features() 得到 128D 特征
            # 在控制台输出图片信息
            logging.info("%-40s %-20s", "正在读的人脸图像 / Reading image:", path_face_personX + "/" + photos_list[i])
            features_128d = return_128d_features(path_face_personX + "/" + photos_list[i])
            # 遇到没有检测出人脸的图片跳过
            if features_128d == 0:
                i += 1
            else:
                features_list_personX.append(features_128d)
    else:
        logging.warning("文件夹内图像文件为空 / Warning: No images in%s/", path_face_personX)

    # 计算 128D 特征的均值
    # personX 的 N 张图像 x 128D -> 1 x 128D
    if features_list_personX:
        features_mean_personX = np.array(features_list_personX, dtype=object).mean(axis=0)
    else:
        features_mean_personX = np.zeros(128, dtype=int, order='F')
    return features_mean_personX


def main():
    logging.basicConfig(level=logging.INFO)
    # 获取已录入的最后一个人脸序号
    person_list = os.listdir("data/data_face/")
    print(person_list)  # ['person_1', 'person_4']

    # 创建并打开features_all.csv文件，将训练的数据写入csv文件
    with open("data/features_all.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for person in person_list:
            # 训练并获取在 data里面的person_的图片，将返回的数据写入csv中
            logging.info("%sperson_%s", path_images_from_camera, person)
            features_mean_personX = return_features_mean_personX(path_images_from_camera + person)

            if len(person.split('_', 2)) == 2:
                # "person_x"
                person_name = person
            else:
                # "person_x_tom"
                person_name = person.split('_', 2)[-1]  # 大于-1的值
            features_mean_personX = np.insert(features_mean_personX, 0, person_name, axis=0)
            # features_mean_personX.csv文件里将会有129个数据，就是person_name+128位特征数据
            writer.writerow(features_mean_personX)
        logging.info("所有录入人脸数据存入：data/features_all.csv")


if __name__ == '__main__':
    main()
