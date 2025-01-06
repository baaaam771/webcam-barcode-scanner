import torch
import cv2
import numpy as np
from pyzbar import pyzbar

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
width, height = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

if not cap.isOpened():
    print("Error: Unable to open webcam.")
    exit()

def calculate_sharpness(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

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

        # Crop the detected region
        cropped_img = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

        # Barcode detection
        barcodes = pyzbar.decode(gray)
        for barcode in barcodes:
            data = barcode.data.decode("utf-8")
            print(f"Detected barcode data: {data}")

        # Sharpness
        sharpness = calculate_sharpness(gray)
        print(f"Sharpness: {sharpness}")

        # Contrast adjustment
        alpha = 1.5  # Contrast control
        beta = 0    # Brightness control
        contrast_img = cv2.convertScaleAbs(cropped_img, alpha=alpha, beta=beta)

        # Gaussian Blur
        blurred_img = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny Edge Detection
        edges = cv2.Canny(blurred_img, 50, 150)

        # Line Segment Detection
        lsd = cv2.createLineSegmentDetector()
        lines = lsd.detect(edges)[0]  # Position of lines
        if lines is not None:
            for line in lines:
                x0, y0, x1, y1 = map(int, line[0])
                cv2.line(cropped_img, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # Inpainting
        mask = cv2.threshold(blurred_img, 1, 255, cv2.THRESH_BINARY_INV)[1]
        inpainted_img = cv2.inpaint(cropped_img, mask, 3, cv2.INPAINT_TELEA)

        # Display results
        cv2.imshow("Original Crop", cropped_img)
        cv2.imshow("Contrast Adjustment", contrast_img)
        cv2.imshow("Blurred", blurred_img)
        cv2.imshow("Edges", edges)
        cv2.imshow("Inpainted", inpainted_img)

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
