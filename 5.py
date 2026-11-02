import cv2
from ultralytics import YOLO
import face_recognition
import os
from mtcnn import MTCNN
from flask import Flask, Response, jsonify, request
import io
from PIL import Image
import numpy as np

# 加载 YOLOv8 模型
model = YOLO('yolov8n.pt')

# 初始化 MTCNN 检测器
detector = MTCNN()

# 加载班级人脸数据（初始为空）
known_face_encodings = []
known_face_names = []

def load_face_data(selected_folder):
    global known_face_encodings, known_face_names
    # 清空之前的编码和名称列表
    known_face_encodings = []
    known_face_names = []
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

def detect_and_recognize(frame, tolerance=0.5):
    # 使用 YOLOv8 模型进行预测
    results = model(frame)

    # 获取检测结果
    person_count = 0
    detection_results = []
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
                        if face_distances[best_match_index] <= tolerance:
                            name = known_face_names[best_match_index]
                        else:
                            name = "Unknown"

                        result_str = f"检测结果: {name}"
                        detection_results.append(result_str)

                        # 在人脸框上显示识别结果
                        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame, detection_results

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    # 获取请求中的班级文件夹名称
    selected_folder = request.form.get('class_folder')
    if not selected_folder:
        return jsonify({"error": "Missing class_folder parameter"}), 400

    # 加载人脸数据
    load_face_data(selected_folder)

    # 获取请求中的图像文件
    file = request.files['image']
    if not file:
        return jsonify({"error": "Missing image file"}), 400

    # 读取图像文件
    img = Image.open(io.BytesIO(file.read()))
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 进行人脸检测和识别
    frame, results = detect_and_recognize(frame)

    # 将处理后的图像转换为字节流
    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()

    # 返回检测结果和处理后的图像
    return Response(frame_bytes, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)