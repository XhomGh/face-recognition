import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import csv
from PIL import Image, ImageDraw, ImageFont

# Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()
# Dlib 人脸 landmark 特征点检测器
predictor = dlib.shape_predictor('./model/shape_predictor_68_face_landmarks.dat')
# Dlib Resnet 人脸识别模型, 提取 128D 的特征矢量
face_reco_model = dlib.face_recognition_model_v1("./model/dlib_face_recognition_resnet_model_v1.dat")

font_path="./font/MSYH.TTC"
font_chinese = ImageFont.truetype(font_path, 30)


class FaceRecognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        # 对帧计数
        self.frame_cnt = 0

        # 用来存放所有录入人脸特征的数组
        self.face_features_known_list = []
        # 存储录入人脸名字
        self.face_name_known_list = []

        # 用来存储上一帧和当前帧 ROI 的质心坐标
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []

        # 用来存储上一帧和当前帧检测出目标的名字
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []

        # 上一帧和当前帧中人脸数的计数器
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0

        # 用来存放进行识别时候对比的欧氏距离
        self.current_frame_face_X_e_distance_list = []

        # 存储当前摄像头中捕获到的所有人脸的坐标名字
        self.current_frame_face_position_list = []
        # 存储当前摄像头中捕获到的人脸特征
        self.current_frame_face_feature_list = []

        # 当前帧和上一帧质心之间的欧式距离
        self.last_current_frame_centroid_e_distance = 0

        # 控制再识别的后续帧数
        # 如果识别出 "unknown" 的脸, 将在 reclassify_interval_cnt 计数到 reclassify_interval 后, 对于人脸进行重新识别
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10
        # 设置视频中的中文
        self.font = cv2.FONT_ITALIC
        self.font_chinese = ImageFont.truetype("./font/MSYH.TTC", 30)

    # 从 "features_all.csv" 读取录入人脸特征
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                # 从1开始到129结束是0处是图像名
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_features_known_list.append(features_someone_arr)
            logging.info("Faces in Database： %d", len(self.face_features_known_list))
            return 1
        else:
            logging.warning("'features_all.csv' not found!")
            logging.warning("Please run 'get_faces_from_camera.py' "
                            "and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'")
            return 0

    def update_fps(self):
        now = time.time()
        # 每秒刷新 fps / Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    @staticmethod
    # 计算两个128D向量间的欧式距离  关键步骤
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # 使用中心追踪来识别人脸
    def centroid_tracker(self):
        # current_frame_face_centroid_list当前帧
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            # 对于当前帧中的人脸1, 和上一帧中的 人脸1/2/3/4/.. 进行欧氏距离计算
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])

                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    # 生成的 cv2 window 上标注信息
    def draw_note(self, img_rd):
        # 添加说明
        # cv2.putText(img_rd, "Face Recognizer with OT", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Frame: " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0),
        1,cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        # for i in range(len(self.current_frame_face_name_list)):
        #     img_rd = cv2.putText(img_rd, "Face_" + str(i + 1), tuple(
        #         [int(self.current_frame_face_centroid_list[i][0]), int(self.current_frame_face_centroid_list[i][1])]),
        #                          self.font,
        #                          0.8, (255, 190, 0),
        #                          1,
        #                          cv2.LINE_AA)

    def draw_name(self, img_rd):
        # 在人脸框下面显示人脸名字
        img = Image.fromarray(cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for i in range(self.current_frame_face_cnt):
            # cv2.putText(img_rd, self.current_frame_face_name_list[i], self.current_frame_face_name_position_list[
            # i], self.font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
            draw.text(xy=self.current_frame_face_position_list[i], text=self.current_frame_face_name_list[i],
                      font=self.font_chinese,
                      fill=(255, 255, 0))
            img_rd = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img_rd

    # 使用中文名字
    def show_chinese_name(self):
        if self.current_frame_face_cnt >= 1:
            # 修改录入的人脸姓名
            self.face_name_known_list[0] = '路遥'.encode('utf-8').decode()
            self.face_name_known_list[1] = '小许'.encode('utf-8').decode()
            # self.face_name_known_list[2] = '坤坤'.encode('utf-8').decode()

    # 处理获取的视频流, 进行人脸识别
    def process(self, stream):
        # 1. 读取存放所有人脸特征的 csv
        if self.get_face_database():
            while stream.isOpened():
                self.frame_cnt += 1
                logging.info("Frame " + str(self.frame_cnt) + " starts")
                flag, img_rd = stream.read()  # 在视频中获取img_rd
                stdin = cv2.waitKey(1)

                # 2. 检测人脸
                faces = detector(img_rd, 0)

                # 3. 更新人脸计数器
                self.last_frame_face_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(faces)

                # 4. 更新上一帧中的人脸列表
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]

                # 5. 更新上一帧和当前帧的质心列表
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []

                # 重要！！！！！
                # 6.1 如果当前帧和上一帧人脸数没有变化
                if (self.current_frame_face_cnt == self.last_frame_face_cnt) and (
                        self.reclassify_interval_cnt != self.reclassify_interval):
                    logging.info("scene 1: 当前帧和上一帧相比没有发生人脸数变化")

                    self.current_frame_face_position_list = []

                    if "unknown" in self.current_frame_face_name_list:
                        logging.info("有未知人脸, 开始进行 reclassify_interval_cnt 计数")
                        self.reclassify_interval_cnt += 1
                    # 关键代码，此处就是关于重心距离的关系
                    if self.current_frame_face_cnt != 0:
                        for k, d in enumerate(faces):
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                            self.current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            img_rd = cv2.rectangle(img_rd,
                                                   tuple([d.left(), d.top()]),
                                                   tuple([d.right(), d.bottom()]),
                                                   (255, 255, 255), 2)

                    # 如果当前帧中有多个人脸, 使用质心追踪
                    if self.current_frame_face_cnt != 1:
                        self.centroid_tracker()

                    # 6.2 在新的人脸下面写入名字
                    for i in range(self.current_frame_face_cnt):
                        img_rd = self.draw_name(img_rd)

                    self.show_chinese_name()
                    self.draw_note(img_rd)

                # 增加一些人脸增加或者减少的细节判断
                # 6.2 如果当前帧和上一帧人脸数发生变化
                else:
                    logging.info("scene 2: 当前帧和上一帧相比人脸数发生变化")
                    self.current_frame_face_position_list = []
                    self.current_frame_face_X_e_distance_list = []
                    self.current_frame_face_feature_list = []
                    self.reclassify_interval_cnt = 0

                    # 6.2.1 人脸数减少 1->0, 2->1, ...
                    if self.current_frame_face_cnt == 0:
                        logging.info("scene 2.1 人脸消失, 当前帧中没有人脸！")
                        # 把名字清除
                        self.current_frame_face_name_list = []
                    # 6.2.2 人脸数增加 0->1, 0->2, ..., 1->2, ...
                    else:
                        logging.info("scene 2.2 出现人脸, 进行人脸识别==============")
                        self.current_frame_face_name_list = []
                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.current_frame_face_feature_list.append(
                                face_reco_model.compute_face_descriptor(img_rd, shape))
                            self.current_frame_face_name_list.append("unknown")

                        # 6.2.2.1 遍历捕获到的图像中所有的人脸
                        for k in range(len(faces)):
                            logging.info("  For face %d in current frame:", k + 1)
                            self.current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            self.current_frame_face_X_e_distance_list = []

                            # 6.2.2.2 每个捕获人脸的名字坐标
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                            # 6.2.2.3 对于某张人脸, 遍历所有存储的人脸特征
                            for i in range(len(self.face_features_known_list)):
                                # 如果 q 数据不为空
                                if str(self.face_features_known_list[i][0]) != '0.0':
                                    e_distance_tmp = self.return_euclidean_distance(
                                        self.current_frame_face_feature_list[k],
                                        self.face_features_known_list[i])
                                    logging.info("with person %d, the e-distance: %f", i + 1, e_distance_tmp)
                                    self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                                else:
                                    # 空数据 person_X
                                    self.current_frame_face_X_e_distance_list.append(7777777)

                            # 6.2.2.4 寻找出最小的欧式距离匹配
                            similar_person_num = self.current_frame_face_X_e_distance_list.index(
                                min(self.current_frame_face_X_e_distance_list))

                            if min(self.current_frame_face_X_e_distance_list) < 0.4:
                                self.current_frame_face_name_list[k] = self.face_name_known_list[similar_person_num]
                                logging.info("  Face recognition result: %s",
                                              self.face_name_known_list[similar_person_num])
                            else:
                                logging.info("Face recognition result: Unknown person")

                        # 7. 生成的窗口添加说明文字
                        self.draw_note(img_rd)

                        # cv2.imwrite("info/info_" + str(self.frame_cnt) + ".png", img_rd) # Dump current frame
                        # image if needed

                # 8. 按下 'q' 键退出
                if stdin == ord('q'):
                    break

                self.update_fps()
                cv2.namedWindow("camera", 1)
                cv2.imshow("camera", img_rd)

    def run(self):
        cap = cv2.VideoCapture("./video/cxk2.mp4")  # 从视频文件获取视频流
        # cap = cv2.VideoCapture(0)  # 获取摄像机视频流，传参后命名为stream

        # frame = np.matrix(cap.read())
        # success, frame = cap.read()
        # arr = np.array(frame)
        # print(len(arr[0]))
        # 将一帧的数据转化为数组后保存到frame.csv中查看
        # with open('./data/frame.csv', 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(arr)

        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()


def main():
    # logging.basicConfig(level=logging.info) # Set log level to 'logging.info' to print info info of every frame
    logging.basicConfig(level=logging.INFO)
    face_recognizer_con = FaceRecognizer()
    face_recognizer_con.run()


if __name__ == '__main__':
    main()
