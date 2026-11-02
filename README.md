# 人脸识别系统 

> 一个用于教室点名/识别人脸的系统：预先从班级照片库离线提取每个人的 128 维人脸编码，检测时先用 **YOLOv8** 定位画面中的"人"，再用 **MTCNN** 在人区域精确定位"脸"，最后用 **face_recognition**（dlib 人脸嵌入）把当前人脸编码与库中已知编码做欧氏距离匹配，输出"姓名 + 人数统计"，并提供 Flask Web、Tkinter、PyQt5 三种界面。

---

## 目录

1. [项目概述](#一项目概述)
2. [技术栈](#二技术栈)
3. [项目结构](#三项目结构)
4. [核心流程与架构](#四核心流程与架构)
5. [四个代码文件的差异](#五四个代码文件的差异)
6. [核心算法原理（面试必问）](#六核心算法原理面试必问)
7. [技术细节与踩坑点](#七技术细节与踩坑点)
8. [可优化方向（加分项）](#八可优化方向加分项)
9. [常见面试题速答](#九常见面试题速答)

---

## 一、项目概述

- **一句话概括**：教室人脸识别系统，核心能力是"**检测出画面里的人 → 定位出每个人的脸 → 认出是谁**"。
- **输入源**：本地视频（1.mp4 / 102.mp4）、摄像头、上传图片。
- **输出**：绘制了绿色边界框和姓名的画面，同时统计人数（Person Count）。
- **离线注册**：从 `class_faces` 班级文件夹的照片中提取人脸编码，构成已知人脸库。
- **三种界面**：Flask Web API、Tkinter 桌面 GUI、PyQt5 摄像头实时界面。

---

## 二、技术栈

| 组件 | 作用 | 版本/要点 |
|---|---|---|
| Python | 开发语言 | 3.x |
| OpenCV (cv2) | 图像读写、绘制框、图像预处理 | 高斯模糊、CLAHE、resize、imshow |
| ultralytics (YOLOv8) | 人体检测 | `yolov8n.pt`（nano 最小模型），COCO class 0 = person |
| MTCNN | 人脸检测 | 三级级联卷积网络（P-Net / R-Net / O-Net） |
| face_recognition | 人脸编码 + 距离匹配 | 基于 dlib 的 128 维人脸嵌入 |
| dlib | 底层人脸特征提取 | 128-d 向量 |
| Flask / Tkinter / PyQt5 | 三种界面 | Web API / 桌面 GUI / 摄像头实时 |

---

## 三、项目结构

```
人脸识别/
├── class_faces/                  # 人脸注册数据（按班级分文件夹）
│   ├── 商务英语/
│   │   └── 郭瑞莹.jpg
│   └── 计算机科学与技术专升本6班/
│       ├── 俞能耀.jpg
│       ├── 叶宇超.jpg
│       ├── 张文杰.jpg
│       ├── 汪翰.jpg
│       ├── 王靖.jpg
│       └── 鱼哎呦.jpg
├── 1.mp4                          # 测试视频（7.py 使用）
├── 102.mp4                        # 测试视频（6.py 使用）
├── yolov8n.pt                     # YOLOv8 预训练权重（nano）
├── 5.py                           # Flask Web API 版本
├── 6.py                           # Tkinter + Canvas 版本
├── 7.py                           # Tkinter + cv2.imshow 版本
└── pyqt.py                        # PyQt5 摄像头实时版本
```

> 数据：`class_faces` 存放学生的照片，文件名即姓名；`*.mp4` 是演示/测试视频；`yolov8n.pt` 是 YOLOv8 的官方预训练权重。

---

## 四、核心流程与架构

### ① 离线人脸注册（每次启动/切换班级时执行一次）

```
class_faces/班级/学生照片.jpg
        ↓  face_recognition.load_image_file(image)
        ↓  face_recognition.face_encodings(image)[0]
  128 维人脸编码
        ↓  按班级拼接文件名
known_face_encodings[]  +  known_face_names[]
   (编码向量列表)            ("班级-学生名")
```

- 遍历班级文件夹下所有图片，逐一提取第一个人脸的 128 维编码。
- 用 `try/except IndexError` 处理"照片里没有检测到人脸"的情况，避免程序崩溃。
- 名称格式：`班级-学生名`，如 `计算机科学与技术专升本6班-俞能耀`。

### ② 每帧实时推理主流程

```
输入帧（视频 / 摄像头 / 上传图片）
   │
   ▼
① 人体检测：YOLOv8  →  得到人框 (x1, y1, x2, y2)  【COCO class_id == 0】
   │
   ▼
② 边界框收缩：padding = 30，人框内缩，去掉多余背景
   │
   ▼
③ 预处理：高斯模糊(5×5) 去噪 → 转灰度 → CLAHE 自适应直方图均衡化 增强对比度
   │
   ▼
④ 人脸检测：MTCNN 在裁剪出的人区域里定位人脸，返回人脸框
   │
   ▼
⑤ 人脸编码：face_recognition.face_encodings → 128 维向量
   │
   ▼
⑥ 距离匹配：face_distance 计算欧氏距离 → argmin 取最近 → 距离 ≤ tolerance 判匹配
   │
   ▼
⑦ 输出：画绿框 + 姓名 + 人数统计（Person Count）
```

**匹配逻辑：**

```python
face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
best_match_index = face_distances.argmin()          # 最近的已知人脸
if face_distances[best_match_index] <= tolerance:   # tolerance = 0.5 / 0.6
    name = known_face_names[best_match_index]
else:
    name = "Unknown"
```

---

## 五、四个代码文件的差异

| 文件 | 界面 | 输入源 | 特点 |
|---|---|---|---|
| **5.py** | Flask Web API | 上传图片（POST /detect） | 前后端分离，接口化部署；返回带框 JPEG |
| **6.py** | Tkinter + Canvas | 视频 102.mp4 | 桌面 GUI，结果在 Label 显示；手动 `del` 释放中间变量 |
| **7.py** | Tkinter + cv2.imshow | 视频 1.mp4 | 独立 OpenCV 窗口；打印匹配距离用于调试 |
| **pyqt.py** | PyQt5 + QTimer | **摄像头**（VideoCapture(0)） | 每 30ms 刷新；结果去重记录；有"结束检测"按钮 |

**5.py 接口说明：**
- 请求：`POST /detect`，表单字段 `class_folder`（班级名）+ `image`（图片文件）。
- 处理：加载班级人脸 → 检测识别 → 用 `cv2.imencode` 编码为 JPEG 字节流返回。
- 错误处理：缺少参数返回 400 + jsonify 错误信息。

**pyqt.py 设计要点：**
- `QTimer` 每 30ms 触发 `update_frame`，定时读摄像头帧。
- 用 `self.shown_results`（set）对识别结果去重，避免同一结果反复刷屏。
- `rgbSwapped()` 处理 BGR→RGB 色彩空间转换。
- `stop_detection` 释放摄像头、停止定时器、清空界面。

---



## 附：如何在本地运行

1. 安装依赖：

```bash
pip install opencv-python ultralytics face_recognition mtcnn flask PyQt5
```

2. 运行不同版本：

```bash
python 5.py     # Flask Web 服务（POST 图片到 /detect）
python 6.py     # Tkinter 界面，播放 102.mp4
python 7.py     # Tkinter 界面，播放 1.mp4
python pyqt.py  # PyQt5 界面，调用摄像头（需电脑有摄像头）
```

> 说明：`face_recognition` 依赖 dlib，Windows 上安装可能需要预编译 wheel 或 Visual Studio C++ 环境；若摄像头不可用，pyqt.py 可改用视频文件测试。
