import os
import cv2
import dlib
import numpy as np

pred_types = {'face': ((0, 17), (0.682, 0.780, 0.909, 0.5)),
              'eyebrow1': ((17, 22), (1.0, 0.498, 0.055, 0.4)),
              'eyebrow2': ((22, 27), (1.0, 0.498, 0.055, 0.4)),
              'nose': ((27, 31), (0.345, 0.239, 0.443, 0.4)),
              'nostril': ((31, 36), (0.345, 0.239, 0.443, 0.4)),
              'eye1': ((36, 42), (0.596, 0.875, 0.541, 0.3)),
              'eye2': ((42, 48), (0.596, 0.875, 0.541, 0.3)),
              'lips': ((48, 60), (0.596, 0.875, 0.541, 0.3)),
              'teeth': ((60, 68), (0.596, 0.875, 0.541, 0.4))
              }


# 绘制直线和关键点
def draw_line(img, shape, i):
    cv2.line(img, pt1=(shape.part(i).x, shape.part(i).y), pt2=(shape.part(i + 1).x, shape.part(i + 1).y),
             color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)


# 连成一圈
def draw_line_circle(img, shape, i, start, end):
    cv2.line(img, pt1=(shape.part(i).x, shape.part(i).y), pt2=(shape.part(i + 1).x, shape.part(i + 1).y),
             color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    cv2.line(img, pt1=(shape.part(start).x, shape.part(start).y), pt2=(shape.part(end).x, shape.part(end).y),
             color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)


# 使用训练好的模型shape_predictor_68_face_landmarks.dat检测出人脸上的68个关键点
def dlib_face_keypoint_detector(img_path, save_result=True):
    # 检测人脸框
    detector = dlib.get_frontal_face_detector()

    # 下载人脸关键点检测模型： http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    predictor_path = '/model/shape_predictor_68_face_landmarks.dat'
    # 检测人脸关键点 载入关键点
    predictor = dlib.shape_predictor(predictor_path)

    img = cv2.imread(img_path)
    print("Processing file: {}".format(img_path))

    # 1 表示图像向上采样一次，图像将被放大一倍，这样可以检测更多的人脸
    dets = detector(img, 1)  # 检测人脸，给出绘制正方形的 高、宽
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, x1, y1, x2, y2))
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)  # 从predictor()函数中获取图片中的人脸关键点
        # print(dir(shape))  # 'num_parts', 'part', 'parts', 'rect'
        print(shape.num_parts)  # 68   打印出关键点的个数
        print(shape.rect)  # 检测到每个面部的矩形框 [(118, 139) (304, 325)]
        print(shape.parts())  # 打印

        # points[(147, 182), (150, 197), (154, 211), (160, 225),...,(222, 227), (215, 228)]   # 68个关键点坐标
        # print(type(shape.part(0)))  # <class 'dlib.point'>
        # 打印出第一个关键点和第2个关键点的坐标
        # print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))

        # 使用np中的matrix()函数将元组转换为68个x,y的矩阵
        landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
        # print(landmarks)

        # 绘制所有的关键点
        for i, point in enumerate(shape.parts()):
            x = point.x
            y = point.y
            print(i)
            cv2.circle(img, (x, y), 1, color=(255, 0, 255), thickness=1, lineType=cv2.CV_16SC3)
            cv2.putText(img, str(i + 1), (x, y), fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=0.3, color=(255, 255, 0))

            # 连接关键点
            if i + 1 < 17:  # face
                draw_line(img, shape, i)
            elif 17 < i + 1 < 22:  # eyebrow1
                draw_line(img, shape, i)
            elif 22 < i + 1 < 27:  # eyebrow2
                draw_line(img, shape, i)
            elif 27 < i + 1 < 31:  # nose
                draw_line(img, shape, i)
            elif 31 < i + 1 < 36:  # nostril
                draw_line(img, shape, i)
            elif 36 < i + 1 < 42:  # eye1
                draw_line_circle(img, shape, i, 36, 42 - 1)
            elif 42 < i + 1 < 48:  # eye2
                draw_line_circle(img, shape, i, 42, 48 - 1)
            elif 48 < i + 1 < 60:  # lips
                draw_line_circle(img, shape, i, 48, 60 - 1)
            elif 60 < i + 1 < 68:  # teeth
                draw_line_circle(img, shape, i, 60, 68 - 1)

    cv2.imshow('detect keypoints', img)
    if save_result:
        dir, filename = os.path.split(img_path)
        save_filename = os.path.join(dir, filename.split('.')[0] + '_keypoint' + '.' + filename.split('.')[1])
        cv2.imwrite(save_filename, img)
    cv2.waitKey(0)


if __name__ == '__main__':
    dlib_face_keypoint_detector('/images/person1.jpg')
    dlib_face_keypoint_detector('/images/person3.jpg')
    dlib_face_keypoint_detector('/images/multi_face.png')
