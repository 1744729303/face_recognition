import cv2
from ultralytics import YOLO
import face_recognition
import os
from mtcnn import MTCNN
import tkinter as tk
from tkinter import StringVar

# 加载 YOLOv8 模型
model = YOLO('yolov8n.pt')  # 可以根据需要选择不同的预训练模型

# 初始化 MTCNN 检测器
detector = MTCNN()

# 创建主窗口
root = tk.Tk()
root.title("人脸识别系统")
root.geometry("800x600")  # 适当增大窗口尺寸

# 获取班级列表
class_folders = [folder for folder in os.listdir('class_faces') if os.path.isdir(os.path.join('class_faces', folder))]
selected_class = StringVar(root)
selected_class.set(class_folders[0])  # 设置默认选项

# 创建下拉菜单
class_menu = tk.OptionMenu(root, selected_class, *class_folders)
class_menu.pack(pady=10)

# 创建用于显示视频帧的 Canvas
video_canvas = tk.Canvas(root, width=640, height=480)
video_canvas.pack()

# 创建用于显示检测结果的 Label
result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

# 加载班级人脸数据（初始为空）
known_face_encodings = []
known_face_names = []

def detect_and_recognize():
    global known_face_encodings, known_face_names
    # 清空之前的编码和名称列表
    known_face_encodings = []
    known_face_names = []
    selected_folder = selected_class.get()
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

    cap = cv2.VideoCapture('102.mp4')
    # 调整视频帧的分辨率
    new_width = 640
    new_height = 480
    # 自定义匹配阈值
    tolerance = 0.6
    while True:
        # 读取视频帧
        ret, frame = cap.read()
        if not ret:
            break

        # 调整帧的分辨率
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

                        print(f"检测到的人脸位置: {face_locations}")
                        print(f"计算出的人脸编码数量: {len(face_encodings)}")

                        for face_encoding in face_encodings:
                            # 计算人脸编码之间的距离
                            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                            best_match_index = face_distances.argmin()
                            if face_distances[best_match_index] <= tolerance:
                                name = known_face_names[best_match_index]
                            else:
                                name = "Unknown"

                            # 在人脸框上显示识别结果
                            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        # 释放人脸检测和识别过程中产生的中间变量
                        del face_image, face_locations, face_encodings

        # 在帧上显示人数
        cv2.putText(frame, f"Person Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 将 OpenCV 的 BGR 图像转换为 RGB 图像，以便在 Tkinter 中显示
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = tk.PhotoImage(data=cv2.imencode('.png', frame_rgb)[1].tobytes())
        video_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        video_canvas.image = photo

        # 更新检测结果显示
        result_label.config(text=f"检测结果: {name}")

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 释放当前帧
        del frame

    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()

# 创建检测按钮
detect_button = tk.Button(root, text="开始检测", command=detect_and_recognize)
detect_button.pack(pady=10)

root.mainloop()