# 화전각도 시각화 (실패)


import torch
import cv2
from pyzbar import pyzbar
from PIL import Image, ImageEnhance
import numpy as np
import math
import datetime

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load YOLOv5 model
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='model.pt').to(device)
print("Model loaded successfully.")

# Open webcam
print("Opening webcam...")
cap = cv2.VideoCapture(0)

# Set webcam resolution
width, height = 1280, 720  # Desired resolution (e.g., 1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Confirm resolution
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Webcam resolution set to: {int(actual_width)}x{int(actual_height)}")

if not cap.isOpened():
    print("Error: Unable to open webcam.")
    exit()

# 로그 파일 설정
log_file = "barcode_log.txt"
def log_barcode_data(barcode_type, data):
    with open(log_file, "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp}, {barcode_type}, {data}\n")

def compute_barcode_orientation(barcode_img):
    """
    바코드의 막대선을 검출하여 가장 많은 선들의 각도를 계산한 후 최적의 회전각을 반환
    """
    gray = cv2.cvtColor(barcode_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Hough 변환을 사용하여 직선 검출
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=5)
    angles = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)
    
    if len(angles) > 0:
        # 히스토그램을 이용하여 가장 많이 검출된 각도를 선택
        hist, bins = np.histogram(angles, bins=np.arange(-90, 91, 5))
        max_angle = bins[np.argmax(hist)]
        return max_angle
    return 0  # 회전이 필요 없는 경우

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Perform YOLOv5 inference
    results = model(frame)
    detections = results.pandas().xyxy[0]

    for i, detection in detections.iterrows():
        x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']]
        x1, y1, x2, y2 = [round(num) for num in [x1, y1, x2, y2]]
        class_id = detection['class']
        confidence = detection['confidence']

        # 바운딩 박스 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Barcode {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 바코드 영역 추출
        cropped_img = frame[y1:y2, x1:x2]
        
        # 바코드 회전 각도 검출
        rotation_angle = compute_barcode_orientation(cropped_img)
        print(f"Detected barcode rotation angle: {rotation_angle} degrees")
        
        # 원형 게이지 그래픽 표시
        center = (x1 + 25, y2 + 80)
        radius = 20
        start_angle = 90
        end_angle = 90 - int(rotation_angle)
        cv2.ellipse(frame, center, (radius, radius), 0, start_angle, end_angle, (255, 0, 0), 2)
        cv2.putText(frame, f"{rotation_angle}°", (x1 + 10, y2 + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 이미지 회전
        rot_matrix = cv2.getRotationMatrix2D((cropped_img.shape[1] // 2, cropped_img.shape[0] // 2), -rotation_angle, 1.0)
        rotated_barcode = cv2.warpAffine(cropped_img, rot_matrix, (cropped_img.shape[1], cropped_img.shape[0]))
        
        # 원본 바코드 디코딩
        ori_barcode_data = ""
        barcodes = pyzbar.decode(cropped_img)
        for barcode in barcodes:
            ori_barcode_data = barcode.data.decode("utf-8")
            log_barcode_data("Original", ori_barcode_data)
        
        # 회전된 바코드 디코딩
        rot_barcode_data = ""
        barcodes = pyzbar.decode(rotated_barcode)
        for barcode in barcodes:
            rot_barcode_data = barcode.data.decode("utf-8")
            log_barcode_data("Rotated", rot_barcode_data)
        
    cv2.imshow('Result', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting program...")
        break

cap.release()
cv2.destroyAllWindows()
print("Resources released. Program terminated.")
