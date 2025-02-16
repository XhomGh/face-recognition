import cv2
import os
import dlib
import glob
import numpy as np


# 计算两个特征向量的欧式距离
# 定义一个计算Euclidean距离的函数
def Eucl_distance(a, b):
    """
    d = 0
    for i in range(len(a)):
    	d += (a[i] - b[i]) * (a[i] - b[i])
    return np.sqrt(d)
    :param a:
    :param b:
    :return:
    """
    return np.linalg.norm(a - b, ord=2)


# 提取人脸的128维特征向量
def extract_face_feature(img_array):
    predictor_path = '/model/shape_predictor_68_face_landmarks.dat'
    face_rec_model_path = '/model/dlib_face_recognition_resnet_model_v1.dat'

    # 导入需要的模型，人脸检测、人脸68个关键点检测模型、人脸识别模型
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)  # 此训练模型专门用来识别是否为同一张脸

    # 当前帧中所有人脸的特征
    # 1 表示图像向上采样一次，图像将被放大一倍，这样可以检测更多的人脸
    dets = detector(img_array, 0)
    print("Number of faces detected: {}".format(len(dets)))

    # 一张图片中可能有多张脸，每张都要验证，同时保存每张人脸的矩形框坐标，用于后续处理
    all_face_feature = []
    all_face_rect = []
    # Now process each face we found.
    for k, d in enumerate(dets):
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, x1, y1, x2, y2))
        all_face_rect.append([x1, y1, x2, y2])
        shape = predictor(img_array, d)
        # Draw the face landmarks on the screen so we can see what face is currently being processed.

        # 把人脸的特征表示成128维向量，然后计算两张人脸的128维向量之间的欧式距离，一般来说如果距离小于0.6认为是同一个人
        # 如果距离大于0.6则认为是不同的人！
        face_descriptor = facerec.compute_face_descriptor(img_array, shape)  # 人脸特征秒描述子
        print("Computing descriptor on aligned image ..")

        # 使用get_face_chip函数生成对齐后的图像
        face_chip = dlib.get_face_chip(img_array, shape)

        # 然后将对齐后图像chip（aligned image）传给api
        face_descriptor_from_prealigned_image = facerec.compute_face_descriptor(face_chip)
        # print(type(face_descriptor_from_prealigned_image))  # <class 'dlib.vector'>
        # print(dir(face_descriptor_from_prealigned_image))  # 'resize', 'set_size', 'shape'
        # print(face_descriptor_from_prealigned_image.shape)  # (128, 1)    表示的是特征向量的值
        # print('face_descriptor_from_prealigned_image:', face_descriptor_from_prealigned_image)
        # 可以把128维的特征向量转换为numpy数组类型
        face_descriptor_from_prealigned_image_np = np.array(face_descriptor_from_prealigned_image)
        all_face_feature.append(face_descriptor_from_prealigned_image_np)
        print(all_face_feature)
    feature_rect = list(zip(all_face_feature, all_face_rect))
    return feature_rect


# 先提取数据库中的特征，当前你也可以把特征提前保存到一个csv文件或这数据库中
def extract_register_face_feature():
    register_face_feature = {}
    img_paths = glob.glob('./pic_images/*.jpg')
    for img_path in img_paths:
        name = os.path.split(img_path)[-1].split('.')[0]
        img = cv2.imread(img_path)
        feature_rect = extract_face_feature(img)
        register_face_feature[name] = feature_rect[0][0]
    return register_face_feature


def main(save_result=True):
    # 注册人脸特征
    all_regiter_feature_dict = extract_register_face_feature()

    # 读取待验证身份的图片
    face_img_paths = glob.glob('./images/kk.jpg')
    for img_path in face_img_paths:
        img_name = os.path.split(img_path)[-1]
        img = cv2.imread(img_path)
        h, w, c = img.shape
        all_feature_rect = extract_face_feature(img)
        for feature_rect in all_feature_rect:
            face_feat = feature_rect[0]
            x1, y1, x2, y2 = feature_rect[1]
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            # cv2.putText(img, str(i+1), (x, y), fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            #             fontScale=0.3, color=(255, 255, 0))
            cv2.putText(img, img_name, (20, h - 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(img, 'Detect Face: {}'.format(len(all_feature_rect)), (20, h - 50),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

            # 当前人脸特征和数据库中每张脸比对以下，并计算距离，然后记录
            all_dist_dict = dict()
            for reg_name, reg_feat in all_regiter_feature_dict.items():
                # print(reg_feat)
                # print(face_feat)
                print(f'src image name: {img_name}, reg image name: {reg_name}')
                dist = Eucl_distance(reg_feat, face_feat)
                print('dist:', dist)
                all_dist_dict[dist] = reg_name
            print('all dist with reg: ', all_dist_dict)
            print(all_dist_dict.keys())
            # print(type(all_dist_dict.keys()[0]))

            print(list(all_dist_dict.keys()))
            print(any(list(all_dist_dict.keys())) < 0.4)
            '''
            >>> any([0.38222860571657313, 0.5497940177651175])<0.4
            False
            # 浮点型返回貌似有点问题，应该返回True才对呀
            '''
            min_dist = min(list(all_dist_dict.keys()))
            if min_dist < 0.4:
                reg_name = all_dist_dict[min_dist]
                if reg_name == "person1" or "kk":
                    # name = "杨幂"
                    reg_name = "KunKun"
                if reg_name == "reba":
                    # name = "迪丽热巴"
                    reg_name = "Reba"
                # cv2.putText(img, 'recog result: {}'.format(reg_name), (50, h-50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #             fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                cv2.putText(img, reg_name, (x1, y1 - 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.8, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)

            else:
                # cv2.putText(img, 'recog result: unknown', (50, h-50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #             fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                cv2.putText(img, 'unknown', (x1, y1 - 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.8, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        if save_result:
            save_name = img_name.split('.')[0] + '_result.' + img_name.split('.')[1]
            save_filename = os.path.join('./pic_images/recog_result', save_name)
            cv2.imwrite(save_filename, img)
        cv2.imshow('detect face', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
