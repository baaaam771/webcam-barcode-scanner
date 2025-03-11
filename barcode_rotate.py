# import torch
# import cv2
# from pyzbar import pyzbar
# from PIL import Image, ImageEnhance
# import numpy as np
# import math

# # Set device
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # Load YOLOv5 model
# print("Loading YOLOv5 model...")
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='model.pt').to(device)
# print("Model loaded successfully.")

# # Open webcam
# print("Opening webcam...")
# cap = cv2.VideoCapture(0)

# # Set webcam resolution
# width, height = 1280, 720  # Desired resolution (e.g., 1280x720)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# # Confirm resolution
# actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print(f"Webcam resolution set to: {int(actual_width)}x{int(actual_height)}")

# if not cap.isOpened():
#     print("Error: Unable to open webcam.")
#     exit()

# def compute_barcode_orientation(barcode_img):
#     """
#     바코드의 막대선을 검출하여 가장 많은 선들의 각도를 계산한 후 최적의 회전각을 반환
#     """
#     gray = cv2.cvtColor(barcode_img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
#     # Hough 변환을 사용하여 직선 검출
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=5)
#     angles = []
    
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
#             angles.append(angle)
    
#     if len(angles) > 0:
#         # 히스토그램을 이용하여 가장 많이 검출된 각도를 선택
#         hist, bins = np.histogram(angles, bins=np.arange(-90, 91, 5))
#         max_angle = bins[np.argmax(hist)]
#         return max_angle
#     return 0  # 회전이 필요 없는 경우

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to grab frame.")
#         break

#     # Perform YOLOv5 inference
#     results = model(frame)
#     detections = results.pandas().xyxy[0]

#     for i, detection in detections.iterrows():
#         x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']]
#         x1, y1, x2, y2 = [round(num) for num in [x1, y1, x2, y2]]
#         class_id = detection['class']
#         confidence = detection['confidence']

#         # 바운딩 박스 그리기
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         label = f"Barcode {confidence:.2f}"
#         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # 바코드 영역 추출
#         cropped_img = frame[y1:y2, x1:x2]
        
#         # 바코드 회전 각도 검출
#         rotation_angle = compute_barcode_orientation(cropped_img)
#         print(f"Detected barcode rotation angle: {rotation_angle} degrees")
        
#         # 이미지 회전
#         center = ((x1 + x2) // 2, (y1 + y2) // 2)
#         rot_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
#         rotated_frame = cv2.warpAffine(frame, rot_matrix, (frame.shape[1], frame.shape[0]))

#         # 바코드 검출 및 디코딩
#         barcodes = pyzbar.decode(rotated_frame[y1:y2, x1:x2])
#         for barcode in barcodes:
#             data = barcode.data.decode("utf-8")
#             print(f"Detected barcode data: {data}")
#             cv2.putText(frame, data, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     cv2.imshow('Result', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print("Exiting program...")
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("Resources released. Program terminated.")

# import torch
# import cv2
# from pyzbar import pyzbar
# from PIL import Image, ImageEnhance
# import numpy as np
# import math

# # Set device
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # Load YOLOv5 model
# print("Loading YOLOv5 model...")
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='model.pt').to(device)
# print("Model loaded successfully.")

# # Open webcam
# print("Opening webcam...")
# cap = cv2.VideoCapture(0)

# # Set webcam resolution
# width, height = 1280, 720  # Desired resolution (e.g., 1280x720)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# # Confirm resolution
# actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print(f"Webcam resolution set to: {int(actual_width)}x{int(actual_height)}")

# if not cap.isOpened():
#     print("Error: Unable to open webcam.")
#     exit()

# def compute_barcode_orientation(barcode_img):
#     """
#     바코드의 막대선을 검출하여 가장 많은 선들의 각도를 계산한 후 최적의 회전각을 반환
#     """
#     gray = cv2.cvtColor(barcode_img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
#     # Hough 변환을 사용하여 직선 검출
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=5)
#     angles = []
    
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
#             angles.append(angle)
    
#     if len(angles) > 0:
#         # 히스토그램을 이용하여 가장 많이 검출된 각도를 선택
#         hist, bins = np.histogram(angles, bins=np.arange(-90, 91, 5))
#         max_angle = bins[np.argmax(hist)]
#         return max_angle
#     return 0  # 회전이 필요 없는 경우

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to grab frame.")
#         break

#     # Perform YOLOv5 inference
#     results = model(frame)
#     detections = results.pandas().xyxy[0]

#     for i, detection in detections.iterrows():
#         x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']]
#         x1, y1, x2, y2 = [round(num) for num in [x1, y1, x2, y2]]
#         class_id = detection['class']
#         confidence = detection['confidence']

#         # 바운딩 박스 그리기
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         label = f"Barcode {confidence:.2f}"
#         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # 바코드 영역 추출
#         cropped_img = frame[y1:y2, x1:x2]
        
#         # 바코드 회전 각도 검출
#         rotation_angle = compute_barcode_orientation(cropped_img)
#         print(f"Detected barcode rotation angle: {rotation_angle} degrees")
        
#         # 이미지 회전
#         center = ((x1 + x2) // 2, (y1 + y2) // 2)
#         rot_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
#         rotated_frame = cv2.warpAffine(frame, rot_matrix, (frame.shape[1], frame.shape[0]))
#         rotated_barcode = cv2.warpAffine(cropped_img, rot_matrix, (cropped_img.shape[1], cropped_img.shape[0]))
        
#         # 바코드 검출 및 디코딩
#         barcodes = pyzbar.decode(rotated_barcode)
#         for barcode in barcodes:
#             data = barcode.data.decode("utf-8")
#             print(f"Detected barcode data: {data}")
#             cv2.putText(frame, data, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # 원본 바운딩 박스와 회전된 바코드 출력
#         combined_view = np.hstack((cropped_img, rotated_barcode))
#         combined_resized = cv2.resize(combined_view, (640, 320))
#         cv2.imshow('Original (Left) | Rotated (Right)', combined_resized)

#     cv2.imshow('Result', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print("Exiting program...")
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("Resources released. Program terminated.")



## best
# import torch
# import cv2
# from pyzbar import pyzbar
# from PIL import Image, ImageEnhance
# import numpy as np
# import math

# # Set device
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # Load YOLOv5 model
# print("Loading YOLOv5 model...")
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='model.pt').to(device)
# print("Model loaded successfully.")

# # Open webcam
# print("Opening webcam...")
# cap = cv2.VideoCapture(0)

# # Set webcam resolution
# width, height = 1280, 720  # Desired resolution (e.g., 1280x720)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# # Confirm resolution
# actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print(f"Webcam resolution set to: {int(actual_width)}x{int(actual_height)}")

# if not cap.isOpened():
#     print("Error: Unable to open webcam.")
#     exit()

# def compute_barcode_orientation(barcode_img):
#     """
#     바코드의 막대선을 검출하여 가장 많은 선들의 각도를 계산한 후 최적의 회전각을 반환
#     """
#     gray = cv2.cvtColor(barcode_img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
#     # Hough 변환을 사용하여 직선 검출
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=5)
#     angles = []
    
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
#             angles.append(angle)
    
#     if len(angles) > 0:
#         # 히스토그램을 이용하여 가장 많이 검출된 각도를 선택
#         hist, bins = np.histogram(angles, bins=np.arange(-90, 91, 5))
#         max_angle = bins[np.argmax(hist)]
#         return max_angle
#     return 0  # 회전이 필요 없는 경우

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to grab frame.")
#         break

#     # Perform YOLOv5 inference
#     results = model(frame)
#     detections = results.pandas().xyxy[0]

#     for i, detection in detections.iterrows():
#         x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']]
#         x1, y1, x2, y2 = [round(num) for num in [x1, y1, x2, y2]]
#         class_id = detection['class']
#         confidence = detection['confidence']

#         # 바운딩 박스 그리기
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         label = f"Barcode {confidence:.2f}"
#         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # 바코드 영역 추출
#         cropped_img = frame[y1:y2, x1:x2]
        
#         # 바코드 회전 각도 검출
#         rotation_angle = compute_barcode_orientation(cropped_img)
#         print(f"Detected barcode rotation angle: {rotation_angle} degrees")
        
#         # 이미지 회전
#         rot_matrix = cv2.getRotationMatrix2D(((x2 - x1) // 2, (y2 - y1) // 2), -rotation_angle, 1.0)
#         rotated_barcode = cv2.warpAffine(cropped_img, rot_matrix, (cropped_img.shape[1], cropped_img.shape[0]))
        
#         # 바코드 검출 및 디코딩
#         barcodes = pyzbar.decode(rotated_barcode)
#         for barcode in barcodes:
#             data = barcode.data.decode("utf-8")
#             print(f"Detected barcode data: {data}")
#             cv2.putText(frame, data, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # 원본 바코드와 회전된 바코드를 개별적으로 출력
#         cv2.imshow('Original Barcode', cropped_img)
#         cv2.imshow('Rotated Barcode', rotated_barcode)

#     cv2.imshow('Result', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print("Exiting program...")
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("Resources released. Program terminated.")


# import torch
# import cv2
# from pyzbar import pyzbar
# from PIL import Image, ImageEnhance
# import numpy as np
# import math

# # Set device
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # Load YOLOv5 model
# print("Loading YOLOv5 model...")
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='model.pt').to(device)
# print("Model loaded successfully.")

# # Open webcam
# print("Opening webcam...")
# cap = cv2.VideoCapture(0)

# # Set webcam resolution
# width, height = 1280, 720  # Desired resolution (e.g., 1280x720)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# # Confirm resolution
# actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print(f"Webcam resolution set to: {int(actual_width)}x{int(actual_height)}")

# if not cap.isOpened():
#     print("Error: Unable to open webcam.")
#     exit()

# def compute_barcode_orientation(barcode_img):
#     """
#     바코드의 막대선을 검출하여 가장 많은 선들의 각도를 계산한 후 최적의 회전각을 반환
#     """
#     gray = cv2.cvtColor(barcode_img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
#     # Hough 변환을 사용하여 직선 검출
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=5)
#     angles = []
    
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
#             angles.append(angle)
    
#     if len(angles) > 0:
#         # 히스토그램을 이용하여 가장 많이 검출된 각도를 선택
#         hist, bins = np.histogram(angles, bins=np.arange(-90, 91, 5))
#         max_angle = bins[np.argmax(hist)]
#         return max_angle
#     return 0  # 회전이 필요 없는 경우

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to grab frame.")
#         break

#     # Perform YOLOv5 inference
#     results = model(frame)
#     detections = results.pandas().xyxy[0]

#     for i, detection in detections.iterrows():
#         x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']]
#         x1, y1, x2, y2 = [round(num) for num in [x1, y1, x2, y2]]
#         class_id = detection['class']
#         confidence = detection['confidence']

#         # 바운딩 박스 그리기
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         label = f"Barcode {confidence:.2f}"
#         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # 바코드 영역 추출
#         cropped_img = frame[y1:y2, x1:x2]
        
#         # 바코드 회전 각도 검출
#         rotation_angle = compute_barcode_orientation(cropped_img)
#         print(f"Detected barcode rotation angle: {rotation_angle} degrees")
        
#         # 이미지 회전
#         rot_matrix = cv2.getRotationMatrix2D((cropped_img.shape[1] // 2, cropped_img.shape[0] // 2), -rotation_angle, 1.0)
#         rotated_barcode = cv2.warpAffine(cropped_img, rot_matrix, (cropped_img.shape[1], cropped_img.shape[0]))
        
#         # 바코드 검출 및 디코딩
#         barcodes = pyzbar.decode(rotated_barcode)
#         for barcode in barcodes:
#             data = barcode.data.decode("utf-8")
#             print(f"Detected barcode data: {data}")
#             cv2.putText(rotated_barcode, data, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

#         # 원본 바코드와 회전된 바코드를 개별적으로 중앙 정렬하여 출력
#         max_width = max(cropped_img.shape[1], rotated_barcode.shape[1])
#         max_height = max(cropped_img.shape[0], rotated_barcode.shape[0])
        
#         original_canvas = np.ones((max_height, max_width, 3), dtype=np.uint8) * 255
#         rotated_canvas = np.ones((max_height, max_width, 3), dtype=np.uint8) * 255
        
#         original_canvas[:cropped_img.shape[0], :cropped_img.shape[1]] = cropped_img
#         rotated_canvas[:rotated_barcode.shape[0], :rotated_barcode.shape[1]] = rotated_barcode
        
#         cv2.imshow('Original Barcode', original_canvas)
#         cv2.imshow('Rotated Barcode', rotated_canvas)

#     cv2.imshow('Result', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print("Exiting program...")
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("Resources released. Program terminated.")


# import torch
# import cv2
# from pyzbar import pyzbar
# from PIL import Image, ImageEnhance
# import numpy as np
# import math

# # Set device
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # Load YOLOv5 model
# print("Loading YOLOv5 model...")
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='model.pt').to(device)
# print("Model loaded successfully.")

# # Open webcam
# print("Opening webcam...")
# cap = cv2.VideoCapture(0)

# # Set webcam resolution
# width, height = 1280, 720  # Desired resolution (e.g., 1280x720)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# # Confirm resolution
# actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print(f"Webcam resolution set to: {int(actual_width)}x{int(actual_height)}")

# if not cap.isOpened():
#     print("Error: Unable to open webcam.")
#     exit()

# def compute_barcode_orientation(barcode_img):
#     """
#     바코드의 막대선을 검출하여 가장 많은 선들의 각도를 계산한 후 최적의 회전각을 반환
#     """
#     gray = cv2.cvtColor(barcode_img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
#     # Hough 변환을 사용하여 직선 검출
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=5)
#     angles = []
    
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
#             angles.append(angle)
    
#     if len(angles) > 0:
#         # 히스토그램을 이용하여 가장 많이 검출된 각도를 선택
#         hist, bins = np.histogram(angles, bins=np.arange(-90, 91, 5))
#         max_angle = bins[np.argmax(hist)]
#         return max_angle
#     return 0  # 회전이 필요 없는 경우

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to grab frame.")
#         break

#     # Perform YOLOv5 inference
#     results = model(frame)
#     detections = results.pandas().xyxy[0]

#     for i, detection in detections.iterrows():
#         x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']]
#         x1, y1, x2, y2 = [round(num) for num in [x1, y1, x2, y2]]
#         class_id = detection['class']
#         confidence = detection['confidence']

#         # 바운딩 박스 그리기
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         label = f"Barcode {confidence:.2f}"
#         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # 바코드 영역 추출
#         cropped_img = frame[y1:y2, x1:x2]
        
#         # 원본 바코드 디코딩
#         ori_barcode_data = ""
#         barcodes = pyzbar.decode(cropped_img)
#         for barcode in barcodes:
#             ori_barcode_data = barcode.data.decode("utf-8")
#             print(f"ori_barcode: {ori_barcode_data}")
#             cv2.putText(frame, f"ori_barcode: {ori_barcode_data}", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
#         # 바코드 회전 각도 검출
#         rotation_angle = compute_barcode_orientation(cropped_img)
#         print(f"Detected barcode rotation angle: {rotation_angle} degrees")
        
#         # 이미지 회전
#         rot_matrix = cv2.getRotationMatrix2D((cropped_img.shape[1] // 2, cropped_img.shape[0] // 2), -rotation_angle, 1.0)
#         rotated_barcode = cv2.warpAffine(cropped_img, rot_matrix, (cropped_img.shape[1], cropped_img.shape[0]))
        
#         # 회전된 바코드 디코딩
#         rot_barcode_data = ""
#         barcodes = pyzbar.decode(rotated_barcode)
#         for barcode in barcodes:
#             rot_barcode_data = barcode.data.decode("utf-8")
#             print(f"rot_barcode: {rot_barcode_data}")
#             cv2.putText(frame, f"rot_barcode: {rot_barcode_data}", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # 원본 바코드와 회전된 바코드를 개별적으로 출력
#         cv2.imshow('Original Barcode', cv2.resize(cropped_img, (400, 200)))
#         cv2.imshow('Rotated Barcode', cv2.resize(rotated_barcode, (400, 200)))

#     cv2.imshow('Result', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print("Exiting program...")
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("Resources released. Program terminated.")

# import torch
# import cv2
# from pyzbar import pyzbar
# from PIL import Image, ImageEnhance
# import numpy as np
# import math

# # Set device
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # Load YOLOv5 model
# print("Loading YOLOv5 model...")
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='model.pt').to(device)
# print("Model loaded successfully.")

# # Open webcam
# print("Opening webcam...")
# cap = cv2.VideoCapture(0)

# # Set webcam resolution
# width, height = 1280, 720  # Desired resolution (e.g., 1280x720)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# # Confirm resolution
# actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print(f"Webcam resolution set to: {int(actual_width)}x{int(actual_height)}")

# if not cap.isOpened():
#     print("Error: Unable to open webcam.")
#     exit()

# def compute_barcode_orientation(barcode_img):
#     """
#     바코드의 막대선을 검출하여 가장 많은 선들의 각도를 계산한 후 최적의 회전각을 반환
#     """
#     gray = cv2.cvtColor(barcode_img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
#     # Hough 변환을 사용하여 직선 검출
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=5)
#     angles = []
    
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
#             angles.append(angle)
    
#     if len(angles) > 0:
#         # 히스토그램을 이용하여 가장 많이 검출된 각도를 선택
#         hist, bins = np.histogram(angles, bins=np.arange(-90, 91, 5))
#         max_angle = bins[np.argmax(hist)]
#         return max_angle
#     return 0  # 회전이 필요 없는 경우

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to grab frame.")
#         break

#     # Perform YOLOv5 inference
#     results = model(frame)
#     detections = results.pandas().xyxy[0]

#     for i, detection in detections.iterrows():
#         x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']]
#         x1, y1, x2, y2 = [round(num) for num in [x1, y1, x2, y2]]
#         class_id = detection['class']
#         confidence = detection['confidence']

#         # 바운딩 박스 그리기
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         label = f"Barcode {confidence:.2f}"
#         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # 바코드 영역 추출
#         cropped_img = frame[y1:y2, x1:x2]
        
#         # 원본 바코드 디코딩
#         ori_barcode_data = ""
#         barcodes = pyzbar.decode(cropped_img)
#         for barcode in barcodes:
#             ori_barcode_data = barcode.data.decode("utf-8")
#             print(f"ori_barcode: {ori_barcode_data}")
#             cv2.putText(frame, f"ori_barcode: {ori_barcode_data}", (x1 + 10, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
#         # 바코드 회전 각도 검출
#         rotation_angle = compute_barcode_orientation(cropped_img)
#         print(f"Detected barcode rotation angle: {rotation_angle} degrees")
        
#         # 이미지 회전
#         rot_matrix = cv2.getRotationMatrix2D((cropped_img.shape[1] // 2, cropped_img.shape[0] // 2), -rotation_angle, 1.0)
#         rotated_barcode = cv2.warpAffine(cropped_img, rot_matrix, (cropped_img.shape[1], cropped_img.shape[0]))
        
#         # 회전된 바코드 디코딩
#         rot_barcode_data = ""
#         barcodes = pyzbar.decode(rotated_barcode)
#         for barcode in barcodes:
#             rot_barcode_data = barcode.data.decode("utf-8")
#             print(f"rot_barcode: {rot_barcode_data}")
#             cv2.putText(frame, f"rot_barcode: {rot_barcode_data}", (x1 + 10, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # 원본 바코드와 회전된 바코드를 개별적으로 출력
#         cv2.imshow('Original Barcode', cv2.resize(cropped_img, (400, 200)))
#         cv2.imshow('Rotated Barcode', cv2.resize(rotated_barcode, (400, 200)))

#     cv2.imshow('Result', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print("Exiting program...")
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("Resources released. Program terminated.")


# import torch
# import cv2
# from pyzbar import pyzbar
# from PIL import Image, ImageEnhance
# import numpy as np
# import math

# # Set device
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # Load YOLOv5 model
# print("Loading YOLOv5 model...")
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='model.pt').to(device)
# print("Model loaded successfully.")

# # Open webcam
# print("Opening webcam...")
# cap = cv2.VideoCapture(0)

# # Set webcam resolution
# width, height = 1280, 720  # Desired resolution (e.g., 1280x720)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# # Confirm resolution
# actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print(f"Webcam resolution set to: {int(actual_width)}x{int(actual_height)}")

# if not cap.isOpened():
#     print("Error: Unable to open webcam.")
#     exit()

# def compute_barcode_orientation(barcode_img):
#     """
#     바코드의 막대선을 검출하여 가장 많은 선들의 각도를 계산한 후 최적의 회전각을 반환
#     """
#     gray = cv2.cvtColor(barcode_img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
#     # Hough 변환을 사용하여 직선 검출
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=5)
#     angles = []
    
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
#             angles.append(angle)
    
#     if len(angles) > 0:
#         # 히스토그램을 이용하여 가장 많이 검출된 각도를 선택
#         hist, bins = np.histogram(angles, bins=np.arange(-90, 91, 5))
#         max_angle = bins[np.argmax(hist)]
#         return max_angle
#     return 0  # 회전이 필요 없는 경우

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to grab frame.")
#         break

#     # Perform YOLOv5 inference
#     results = model(frame)
#     detections = results.pandas().xyxy[0]

#     for i, detection in detections.iterrows():
#         x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']]
#         x1, y1, x2, y2 = [round(num) for num in [x1, y1, x2, y2]]
#         class_id = detection['class']
#         confidence = detection['confidence']

#         # 바운딩 박스 그리기
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         label = f"Barcode {confidence:.2f}"
#         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # 바코드 영역 추출
#         cropped_img = frame[y1:y2, x1:x2]
        
#         # 원본 바코드 디코딩
#         ori_barcode_data = ""
#         barcodes = pyzbar.decode(cropped_img)
#         for barcode in barcodes:
#             ori_barcode_data = barcode.data.decode("utf-8")
#             print(f"ori_barcode: {ori_barcode_data}")
#             cv2.putText(frame, f"ori_barcode: {ori_barcode_data}", (x1 + 10, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
#         # 바코드 회전 각도 검출
#         rotation_angle = compute_barcode_orientation(cropped_img)
#         print(f"Detected barcode rotation angle: {rotation_angle} degrees")
        
#         # 이미지 회전
#         rot_matrix = cv2.getRotationMatrix2D((cropped_img.shape[1] // 2, cropped_img.shape[0] // 2), -rotation_angle, 1.0)
#         rotated_barcode = cv2.warpAffine(cropped_img, rot_matrix, (cropped_img.shape[1], cropped_img.shape[0]))
        
#         # 회전된 바코드 디코딩
#         rot_barcode_data = ""
#         barcodes = pyzbar.decode(rotated_barcode)
#         for barcode in barcodes:
#             rot_barcode_data = barcode.data.decode("utf-8")
#             print(f"rot_barcode: {rot_barcode_data}")
#             cv2.putText(frame, f"rot_barcode: {rot_barcode_data}", (x1 + 10, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # 원본 바코드와 회전된 바코드를 개별적으로 출력
#         cv2.imshow('Original Barcode', cv2.resize(cropped_img, (400, 200)))
#         cv2.imshow('Rotated Barcode', cv2.resize(rotated_barcode, (400, 200)))

#     cv2.imshow('Result', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print("Exiting program...")
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("Resources released. Program terminated.")

# import torch
# import cv2
# from pyzbar import pyzbar
# from PIL import Image, ImageEnhance
# import numpy as np
# import math

# # Set device
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # Load YOLOv5 model
# print("Loading YOLOv5 model...")
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='model.pt').to(device)
# print("Model loaded successfully.")

# # Open webcam
# print("Opening webcam...")
# cap = cv2.VideoCapture(0)

# # Set webcam resolution
# width, height = 1280, 720  # Desired resolution (e.g., 1280x720)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# # Confirm resolution
# actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print(f"Webcam resolution set to: {int(actual_width)}x{int(actual_height)}")

# if not cap.isOpened():
#     print("Error: Unable to open webcam.")
#     exit()

# def compute_barcode_orientation(barcode_img):
#     """
#     바코드의 막대선을 검출하여 가장 많은 선들의 각도를 계산한 후 최적의 회전각을 반환
#     """
#     gray = cv2.cvtColor(barcode_img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
#     # Hough 변환을 사용하여 직선 검출
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=5)
#     angles = []
    
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
#             angles.append(angle)
    
#     if len(angles) > 0:
#         # 히스토그램을 이용하여 가장 많이 검출된 각도를 선택
#         hist, bins = np.histogram(angles, bins=np.arange(-90, 91, 5))
#         max_angle = bins[np.argmax(hist)]
#         return max_angle
#     return 0  # 회전이 필요 없는 경우

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to grab frame.")
#         break

#     # Perform YOLOv5 inference
#     results = model(frame)
#     detections = results.pandas().xyxy[0]

#     for i, detection in detections.iterrows():
#         x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']]
#         x1, y1, x2, y2 = [round(num) for num in [x1, y1, x2, y2]]
#         class_id = detection['class']
#         confidence = detection['confidence']

#         # 바운딩 박스 그리기
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         label = f"Barcode {confidence:.2f}"
#         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # 바코드 영역 추출
#         cropped_img = frame[y1:y2, x1:x2]
        
#         # 원본 바코드 디코딩
#         ori_barcode_data = ""
#         barcodes = pyzbar.decode(cropped_img)
#         for barcode in barcodes:
#             ori_barcode_data = barcode.data.decode("utf-8")
#             print(f"ori_barcode: {ori_barcode_data}")
#             cv2.putText(frame, f"ori_barcode: {ori_barcode_data}", (x1 + 10, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
#         # 바코드 회전 각도 검출
#         rotation_angle = compute_barcode_orientation(cropped_img)
#         print(f"Detected barcode rotation angle: {rotation_angle} degrees")
        
#         # 이미지 회전
#         rot_matrix = cv2.getRotationMatrix2D((cropped_img.shape[1] // 2, cropped_img.shape[0] // 2), -rotation_angle, 1.0)
#         rotated_barcode = cv2.warpAffine(cropped_img, rot_matrix, (cropped_img.shape[1], cropped_img.shape[0]))
        
#         # 회전된 바코드 디코딩
#         rot_barcode_data = ""
#         barcodes = pyzbar.decode(rotated_barcode)
#         for barcode in barcodes:
#             rot_barcode_data = barcode.data.decode("utf-8")
#             print(f"rot_barcode: {rot_barcode_data}")
#             cv2.putText(frame, f"rot_barcode: {rot_barcode_data}", (x1 + 10, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#         cv2.putText(frame, f"rotation_angle: {rotation_angle}°", (x1 + 10, y2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # 원본 바코드와 회전된 바코드를 개별적으로 출력
#         cv2.imshow('Original Barcode', cv2.resize(cropped_img, (400, 200)))
#         cv2.imshow('Rotated Barcode', cv2.resize(rotated_barcode, (400, 200)))

#     cv2.imshow('Result', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print("Exiting program...")
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("Resources released. Program terminated.")


# import torch
# import cv2
# from pyzbar import pyzbar
# from PIL import Image, ImageEnhance
# import numpy as np
# import math

# # Set device
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # Load YOLOv5 model
# print("Loading YOLOv5 model...")
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='model.pt').to(device)
# print("Model loaded successfully.")

# # Open webcam
# print("Opening webcam...")
# cap = cv2.VideoCapture(0)

# # Set webcam resolution
# width, height = 1280, 720  # Desired resolution (e.g., 1280x720)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# # Confirm resolution
# actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print(f"Webcam resolution set to: {int(actual_width)}x{int(actual_height)}")

# if not cap.isOpened():
#     print("Error: Unable to open webcam.")
#     exit()

# def compute_barcode_orientation(barcode_img):
#     """
#     바코드의 막대선을 검출하여 가장 많은 선들의 각도를 계산한 후 최적의 회전각을 반환
#     """
#     gray = cv2.cvtColor(barcode_img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
#     # Hough 변환을 사용하여 직선 검출
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=5)
#     angles = []
    
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
#             angles.append(angle)
    
#     if len(angles) > 0:
#         # 히스토그램을 이용하여 가장 많이 검출된 각도를 선택
#         hist, bins = np.histogram(angles, bins=np.arange(-90, 91, 5))
#         max_angle = bins[np.argmax(hist)]
#         return max_angle
#     return 0  # 회전이 필요 없는 경우

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to grab frame.")
#         break

#     # Perform YOLOv5 inference
#     results = model(frame)
#     detections = results.pandas().xyxy[0]

#     for i, detection in detections.iterrows():
#         x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']]
#         x1, y1, x2, y2 = [round(num) for num in [x1, y1, x2, y2]]
#         class_id = detection['class']
#         confidence = detection['confidence']

#         # 바운딩 박스 그리기
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         label = f"Barcode {confidence:.2f}"
#         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # 바코드 영역 추출
#         cropped_img = frame[y1:y2, x1:x2]
        
#         # 원본 바코드 디코딩
#         ori_barcode_data = ""
#         barcodes = pyzbar.decode(cropped_img)
#         for barcode in barcodes:
#             ori_barcode_data = barcode.data.decode("utf-8")
#             print(f"ori_barcode: {ori_barcode_data}")
#             cv2.putText(frame, f"ori_barcode: {ori_barcode_data}", (x1 + 10, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
#         # 바코드 회전 각도 검출
#         rotation_angle = compute_barcode_orientation(cropped_img)
#         print(f"Detected barcode rotation angle: {rotation_angle} degrees")
        
#         # 이미지 회전
#         rot_matrix = cv2.getRotationMatrix2D((cropped_img.shape[1] // 2, cropped_img.shape[0] // 2), -rotation_angle, 1.0)
#         rotated_barcode = cv2.warpAffine(cropped_img, rot_matrix, (cropped_img.shape[1], cropped_img.shape[0]))
        
#         # 회전된 바코드 디코딩
#         rot_barcode_data = ""
#         barcodes = pyzbar.decode(rotated_barcode)
#         for barcode in barcodes:
#             rot_barcode_data = barcode.data.decode("utf-8")
#             print(f"rot_barcode: {rot_barcode_data}")
#             cv2.putText(frame, f"rot_barcode: {rot_barcode_data}", (x1 + 10, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#         cv2.putText(frame, f"rotation_angle: {rotation_angle}", (x1 + 10, y2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # 원본 바코드와 회전된 바코드를 개별적으로 출력
#         cv2.imshow('Original Barcode', cv2.resize(cropped_img, (400, 200)))
#         cv2.imshow('Rotated Barcode', cv2.resize(rotated_barcode, (400, 200)))

#     cv2.imshow('Result', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print("Exiting program...")
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("Resources released. Program terminated.")


# import torch
# import cv2
# from pyzbar import pyzbar
# from PIL import Image, ImageEnhance
# import numpy as np
# import math

# # Set device
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # Load YOLOv5 model
# print("Loading YOLOv5 model...")
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='model.pt').to(device)
# print("Model loaded successfully.")

# # Open webcam
# print("Opening webcam...")
# cap = cv2.VideoCapture(0)

# # Set webcam resolution
# width, height = 1280, 720  # Desired resolution (e.g., 1280x720)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# # Confirm resolution
# actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print(f"Webcam resolution set to: {int(actual_width)}x{int(actual_height)}")

# if not cap.isOpened():
#     print("Error: Unable to open webcam.")
#     exit()

# def compute_barcode_orientation(barcode_img):
#     """
#     바코드의 막대선을 검출하여 가장 많은 선들의 각도를 계산한 후 최적의 회전각을 반환
#     """
#     gray = cv2.cvtColor(barcode_img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
#     # Hough 변환을 사용하여 직선 검출
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=5)
#     angles = []
    
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
#             angles.append(angle)
    
#     if len(angles) > 0:
#         # 히스토그램을 이용하여 가장 많이 검출된 각도를 선택
#         hist, bins = np.histogram(angles, bins=np.arange(-90, 91, 5))
#         max_angle = bins[np.argmax(hist)]
#         return max_angle
#     return 0  # 회전이 필요 없는 경우

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to grab frame.")
#         break

#     # Perform YOLOv5 inference
#     results = model(frame)
#     detections = results.pandas().xyxy[0]

#     for i, detection in detections.iterrows():
#         x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']]
#         x1, y1, x2, y2 = [round(num) for num in [x1, y1, x2, y2]]
#         class_id = detection['class']
#         confidence = detection['confidence']

#         # 바운딩 박스 그리기
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         label = f"Barcode {confidence:.2f}"
#         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # 바코드 영역 추출
#         cropped_img = frame[y1:y2, x1:x2]
        
#         # 원본 바코드 디코딩
#         ori_barcode_data = ""
#         barcodes = pyzbar.decode(cropped_img)
#         for barcode in barcodes:
#             ori_barcode_data = barcode.data.decode("utf-8")
#             print(f"ori_barcode: {ori_barcode_data}")
#             cv2.putText(frame, f"ori_barcode: {ori_barcode_data if ori_barcode_data else ''}", (x1 + 10, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
#         # 바코드 회전 각도 검출
#         rotation_angle = compute_barcode_orientation(cropped_img)
#         print(f"Detected barcode rotation angle: {rotation_angle} degrees")
        
#         # 이미지 회전
#         rot_matrix = cv2.getRotationMatrix2D((cropped_img.shape[1] // 2, cropped_img.shape[0] // 2), -rotation_angle, 1.0)
#         rotated_barcode = cv2.warpAffine(cropped_img, rot_matrix, (cropped_img.shape[1], cropped_img.shape[0]))
        
#         # 회전된 바코드 디코딩
#         rot_barcode_data = ""
#         barcodes = pyzbar.decode(rotated_barcode)
#         for barcode in barcodes:
#             rot_barcode_data = barcode.data.decode("utf-8")
#             print(f"rot_barcode: {rot_barcode_data}")
#             cv2.putText(frame, f"rot_barcode: {rot_barcode_data if rot_barcode_data else ''}", (x1 + 10, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#         cv2.putText(frame, f"rotation_angle: {rotation_angle}", (x1 + 10, y2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # 원본 바코드와 회전된 바코드를 개별적으로 출력
#         cv2.imshow('Original Barcode', cv2.resize(cropped_img, (400, 200)))
#         cv2.imshow('Rotated Barcode', cv2.resize(rotated_barcode, (400, 200)))

#     cv2.imshow('Result', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print("Exiting program...")
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("Resources released. Program terminated.")



import torch
import cv2
from pyzbar import pyzbar
from PIL import Image, ImageEnhance
import numpy as np
import math

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
        
        # 원본 바코드 디코딩
        ori_barcode_data = ""
        barcodes = pyzbar.decode(cropped_img)
        for barcode in barcodes:
            ori_barcode_data = barcode.data.decode("utf-8")
            print(f"ori_barcode: {ori_barcode_data}")
            cv2.putText(frame, f"ori_barcode: {ori_barcode_data}" if ori_barcode_data else "ori_barcode:", (x1 + 10, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 바코드 회전 각도 검출
        rotation_angle = compute_barcode_orientation(cropped_img)
        print(f"Detected barcode rotation angle: {rotation_angle} degrees")
        
        # 이미지 회전
        rot_matrix = cv2.getRotationMatrix2D((cropped_img.shape[1] // 2, cropped_img.shape[0] // 2), -rotation_angle, 1.0)
        rotated_barcode = cv2.warpAffine(cropped_img, rot_matrix, (cropped_img.shape[1], cropped_img.shape[0]))
        
        # 회전된 바코드 디코딩
        rot_barcode_data = ""
        barcodes = pyzbar.decode(rotated_barcode)
        for barcode in barcodes:
            rot_barcode_data = barcode.data.decode("utf-8")
            print(f"rot_barcode: {rot_barcode_data}")
            cv2.putText(frame, f"rot_barcode: {rot_barcode_data}" if rot_barcode_data else "rot_barcode:", (x1 + 10, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, f"rotation_angle: {rotation_angle}", (x1 + 10, y2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 원본 바코드와 회전된 바코드를 개별적으로 출력
        cv2.imshow('Original Barcode', cv2.resize(cropped_img, (400, 200)))
        cv2.imshow('Rotated Barcode', cv2.resize(rotated_barcode, (400, 200)))

    cv2.imshow('Result', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting program...")
        break

cap.release()
cv2.destroyAllWindows()
print("Resources released. Program terminated.")
