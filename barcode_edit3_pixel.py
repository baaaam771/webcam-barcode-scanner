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
width, height = 1280, 720  # Desired resolution
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

        # Calculate bounding box pixel count
        w, h = x2 - x1, y2 - y1
        bounding_box_pixels = w * h
        print(f"Bounding box pixel count: {bounding_box_pixels}")

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Barcode {confidence:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1), (x1 + label_size[0], y1 - label_size[1] - baseline), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Crop and preprocess barcode image
        cropped_img = frame[y1:y2, x1:x2]

        # Apply contrast enhancement
        alpha = 1.5  # Contrast control
        beta = 0    # Brightness control
        contrast_img = cv2.convertScaleAbs(cropped_img, alpha=alpha, beta=beta)

        # Convert to PIL Image for preprocessing
        pil_img = Image.fromarray(cv2.cvtColor(contrast_img, cv2.COLOR_BGR2RGB))

        # Apply sharpness enhancement
        enhancer = ImageEnhance.Sharpness(pil_img)
        sharp_img = enhancer.enhance(2.0)

        # Convert sharp image back to OpenCV format
        sharp_cv_img = cv2.cvtColor(np.array(sharp_img), cv2.COLOR_RGB2BGR)

        # Apply bilateral filter for noise reduction
        bilateral_img = cv2.bilateralFilter(sharp_cv_img, d=9, sigmaColor=75, sigmaSpace=75)

        # Combine all transformations into a single display
        combined = np.hstack((cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB), sharp_cv_img, contrast_img, bilateral_img))

        # Resize for better visibility if needed
        combined_resized = cv2.resize(combined, (1280, 360))
        cv2.imshow('Processing Stages (Original | Contrast | Sharpness | Bilateral Filter)', combined_resized)

        # Decode barcodes using Pyzbar
        barcodes = pyzbar.decode(bilateral_img)
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
