import cv2
from ultralytics import YOLO
import face_recognition
import os
from mtcnn import MTCNN
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QLabel, QPushButton, QTextEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer


# 加载 YOLOv8 模型
model = YOLO('yolov8n.pt')

# 初始化 MTCNN 检测器
detector = MTCNN()

# 加载班级人脸数据（初始为空）
known_face_encodings = []
known_face_names = []


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.shown_results = set()
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.tolerance = 0.5

    def initUI(self):
        self.setWindowTitle("人脸识别系统")
        self.setGeometry(100, 100, 800, 600)

        # 获取班级列表
        class_folders = [folder for folder in os.listdir('class_faces') if os.path.isdir(os.path.join('class_faces', folder))]

        # 创建班级下拉菜单
        self.class_combo = QComboBox(self)
        for folder in class_folders:
            self.class_combo.addItem(folder)

        # 创建用于显示检测结果的文本编辑框
        self.result_text = QTextEdit(self)
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("font-size: 12px;")

        # 创建检测按钮
        self.detect_button = QPushButton("开始检测", self)
        self.detect_button.clicked.connect(self.detect_and_recognize)

        # 创建结束检测按钮
        self.stop_button = QPushButton("结束检测", self)
        self.stop_button.clicked.connect(self.stop_detection)

        # 创建用于显示视频的标签
        self.video_label = QLabel(self)

        # 布局管理
        layout = QVBoxLayout()
        layout.addWidget(self.class_combo)
        layout.addWidget(self.result_text)
        layout.addWidget(self.detect_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.video_label)

        self.setLayout(layout)

    def detect_and_recognize(self):
        global known_face_encodings, known_face_names
        # 清空之前的编码和名称列表
        known_face_encodings = []
        known_face_names = []
        selected_folder = self.class_combo.currentText()
        class_dir = os.path.join('class_faces', selected_folder)
        for student_image in os.listdir(class_dir):
            image_path = os.path.join(class_dir, student_image)
            try:
                image = face_recognition.load_image_file(image_path)
                face_encoding = face_recognition.face_encodings(image)[0]
                known_face_encodings.append(face_encoding)
                student_name = f"{selected_folder}-{os.path.splitext(student_image)[0]}"
                known_face_names.append(student_name)
            except IndexError:
                print(f"未在 {image_path} 中检测到人脸")

        self.cap = cv2.VideoCapture(0)
        # 调整视频帧的分辨率
        new_width = 640
        new_height = 480
        self.result_text.clear()
        self.shown_results = set()
        if self.cap.isOpened():
            self.timer.start(30)  # 每 30 毫秒更新一次帧

    def update_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 调整帧的分辨率
                new_width = 640
                new_height = 480
                frame = cv2.resize(frame, (new_width, new_height))

                # 使用 YOLOv8 模型进行预测
                results = model(frame)

                # 获取检测结果
                person_count = 0
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        class_id = int(box.cls[0])
                        if class_id == 0:  # 在 COCO 数据集中，人的类别 ID 为 0
                            person_count += 1
                            # 获取边界框坐标
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            # 适当缩小边界框范围
                            padding = 30  # 可以根据实际情况调整
                            x1 = max(x1 + padding, 0)
                            y1 = max(y1 + padding, 0)
                            x2 = min(x2 - padding, frame.shape[1])
                            y2 = min(y2 - padding, frame.shape[0])

                            # 检查边界框是否合法
                            if x2 > x1 and y2 > y1:
                                # 在帧上绘制边界框
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                                # 人脸检测和识别
                                face_image = frame[y1:y2, x1:x2]
                                # 高斯模糊去除噪声
                                face_image = cv2.GaussianBlur(face_image, (5, 5), 0)
                                # 自适应直方图均衡化
                                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                                face_image = clahe.apply(face_image)
                                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)

                                # 使用 MTCNN 进行人脸检测
                                results = detector.detect_faces(face_image)
                                face_locations = []
                                for result in results:
                                    x, y, w, h = result['box']
                                    face_locations.append((y, x + w, y + h, x))

                                face_encodings = face_recognition.face_encodings(face_image, face_locations)

                                for face_encoding in face_encodings:
                                    # 计算人脸编码之间的距离
                                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                                    best_match_index = face_distances.argmin()
                                    if face_distances[best_match_index] <= self.tolerance:
                                        name = known_face_names[best_match_index]
                                    else:
                                        name = "Unknown"

                                    result_str = f"检测结果: {name}"
                                    if result_str not in self.shown_results:
                                        self.result_text.append(result_str)
                                        self.shown_results.add(result_str)

                                    # 在人脸框上显示识别结果
                                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # 将 OpenCV 图像转换为 QImage
                height, width, channel = frame.shape
                bytesPerLine = 3 * width
                qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(qImg)
                self.video_label.setPixmap(pixmap)
            else:
                self.timer.stop()
                self.cap.release()
        else:
            self.timer.stop()

    def stop_detection(self):
        if self.cap is not None and self.cap.isOpened():
            self.timer.stop()
            self.cap.release()
            self.video_label.clear()  # 清空显示的视频画面
            self.result_text.clear()  # 清空检测结果


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())