# import torch
# import cv2
# import numpy as np
# from pyzbar import pyzbar

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
# width, height = 1280, 720
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# if not cap.isOpened():
#     print("Error: Unable to open webcam.")
#     exit()

# def calculate_sharpness(image):
#     return cv2.Laplacian(image, cv2.CV_64F).var()

# # 병합을 위한 함수 정의
# def stack_images(images, titles, max_width=3):
#     """
#     이미지를 한 화면에 병합해서 보여줍니다.
#     :param images: 이미지 배열 (list)
#     :param titles: 각 이미지에 해당하는 제목 (list)
#     :param max_width: 가로로 배치할 최대 이미지 수
#     """
#     max_width = min(max_width, len(images))
#     rows = (len(images) + max_width - 1) // max_width  # 행 수 계산
#     cols = min(max_width, len(images))  # 열 수 계산
#     img_h, img_w = images[0].shape[:2]
#     white_img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255  # 빈 이미지
    
#     # 타이틀 추가와 크기 맞추기
#     titled_images = []
#     for img, title in zip(images, titles):
#         if len(img.shape) == 2:  # 흑백 이미지를 컬러로 변환
#             img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#         overlay = img.copy()
#         cv2.putText(overlay, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         titled_images.append(overlay)
    
#     # 이미지를 행렬 형태로 배치
#     grid = []
#     for i in range(rows):
#         row_images = titled_images[i * max_width:(i + 1) * max_width]
#         while len(row_images) < cols:  # 부족한 이미지는 빈 이미지로 채움
#             row_images.append(white_img)
#         grid.append(cv2.hconcat(row_images))
    
#     # 병합된 전체 이미지 반환
#     return cv2.vconcat(grid)

# # Main loop
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

#         # Crop the detected region
#         cropped_img = frame[y1:y2, x1:x2]
#         gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

#         # Barcode detection
#         barcodes = pyzbar.decode(gray)
#         for barcode in barcodes:
#             data = barcode.data.decode("utf-8")
#             print(f"Detected barcode data: {data}")

#         # Sharpness
#         sharpness = calculate_sharpness(gray)
#         print(f"Sharpness: {sharpness}")

#         # Contrast adjustment
#         alpha = 1.5  # Contrast control
#         beta = 0    # Brightness control
#         contrast_img = cv2.convertScaleAbs(cropped_img, alpha=alpha, beta=beta)

#         # Gaussian Blur
#         blurred_img = cv2.GaussianBlur(gray, (3, 3), 0)

#         # Canny Edge Detection
#         edges = cv2.Canny(blurred_img, 50, 150)

#         # Line Segment Detection
#         lsd = cv2.createLineSegmentDetector()
#         lines = lsd.detect(edges)[0]  # Position of lines
#         if lines is not None:
#             for line in lines:
#                 x0, y0, x1, y1 = map(int, line[0])
#                 cv2.line(cropped_img, (x0, y0), (x1, y1), (0, 255, 0), 2)

#         # Inpainting
#         mask = cv2.threshold(blurred_img, 1, 255, cv2.THRESH_BINARY_INV)[1]
#         inpainted_img = cv2.inpaint(cropped_img, mask, 3, cv2.INPAINT_TELEA)

#         # Combine and display results
#         result_images = [cropped_img, contrast_img, blurred_img, edges, inpainted_img]
#         titles = ["Original Crop", "Contrast Adjustment", "Blurred", "Edges", "Inpainted"]
#         merged_image = stack_images(result_images, titles)
#         cv2.imshow("Processed Results", merged_image)

#     # Display the frame
#     cv2.imshow('Result', frame)

#     # Exit condition
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print("Exiting program...")
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
# print("Resources released. Program terminated.")



import torch
import cv2
from pyzbar import pyzbar
from PIL import Image, ImageEnhance
import numpy as np

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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Perform YOLOv5 inference
    results = model(frame)
    detections = results.pandas().xyxy[0]

    # Process each detection
    for i, detection in detections.iterrows():
        # Extract bounding box coordinates
        x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']]
        x1, y1, x2, y2 = [round(num) for num in [x1, y1, x2, y2]]
        class_id = detection['class']
        confidence = detection['confidence']

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Barcode {confidence:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1), (x1 + label_size[0], y1 - label_size[1] - baseline), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Crop and preprocess barcode image
        cropped_img = frame[y1:y2, x1:x2]

        # Apply contrast enhancement
        alpha = 1.5  # Contrast control (1.0-3.0)
        beta = 0    # Brightness control (0-100)
        contrast_img = cv2.convertScaleAbs(cropped_img, alpha=alpha, beta=beta)

        # Convert to PIL Image for preprocessing
        pil_img = Image.fromarray(cv2.cvtColor(contrast_img, cv2.COLOR_BGR2RGB))

        # Apply sharpness enhancement
        enhancer = ImageEnhance.Sharpness(pil_img)
        sharp_img = enhancer.enhance(2.0)  # Increase sharpness by a factor of 2

        # Convert sharp image back to OpenCV format
        sharp_cv_img = cv2.cvtColor(np.array(sharp_img), cv2.COLOR_RGB2BGR)

        # Apply inpainting to remove noise or artifacts
        mask = cv2.inRange(sharp_cv_img, (0, 0, 0), (30, 30, 30))  # Example mask for dark artifacts
        inpainted_img = cv2.inpaint(sharp_cv_img, mask, 3, cv2.INPAINT_TELEA)

        # Combine all transformations into a single display
        combined = np.hstack((cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB),
                              sharp_cv_img, contrast_img, inpainted_img))

        # Resize for better visibility if needed
        combined_resized = cv2.resize(combined, (1280, 360))
        cv2.imshow('Processing Stages (Original | Contrast | Sharpness | Inpainting)', combined_resized)

        # Decode barcodes using Pyzbar
        barcodes = pyzbar.decode(inpainted_img)
        for barcode in barcodes:
            data = barcode.data.decode("utf-8")
            print(f"Detected barcode data: {data}")

            # Display barcode data
            data_size, baseline = cv2.getTextSize(data, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y2), (x1 + data_size[0], y2 + data_size[1]), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, data, (x1, y2 + data_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Display the frame
    cv2.imshow('Result', frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting program...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Resources released. Program terminated.")

