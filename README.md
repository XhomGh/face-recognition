# Face Recognition Project

## 项目结构      
/face_recognition/        
├── docs/               # （可选）存放文档     
├── src/                # （可选）可重构为源码目录  
│   ├── 0_1图片识别先导.py  
│   ├── 0_2opd人脸关键点检测.py      
│   ├── 0_3opd标记.py  
│   ├── 1.从相机获取训练素材.py    
│   ├── 2.获取人脸特征.py    
│   └── 3.相机人脸识别.py  
├── data/               # 空目录（通过.gitignore忽略内容）  
├── images/             # 空目录（同上）  
├── models/             # 空目录（同上）  
├── requirements.txt    # 依赖库列表  
├── README.md           # 项目说明文档  
└── .gitignore          # Git忽略规则  



## 安装依赖
```bash
pip install -r requirements.txt
```
## 使用说明

### 记得安装依赖，如若不然，项目无法启动！因为opencv与python版本严格对应。详见[opencv](https://pypi.tuna.tsinghua.edu.cn/simple/opencv-python/)

### 图片人脸识别
1.init图片识别

```bash
python 0_1图片识别先导.py
```
2.人脸关键点检测：
```bash
python 0_2opd人脸关键点检测.py
```
3.人脸关键点标记：
```bash
python 0_3opd标记.py
```


### 视频人脸识别
1.采集训练素材：
```bash
python 1.从相机获取训练素材.py
```
2.提取人脸特征：
```bash
python 2.获取人脸特征.py
```
3.实时识别：
```bash
python 3.相机人脸识别.py
```
## 注意事项
 - 需自行准备摄像头设备
 - 首次运行前请安装依赖
