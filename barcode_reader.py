# import torch
# import cv2
# from pyzbar import pyzbar 

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = torch.hub.load('ultralytics/yolov5', 'custom',path='model.pt').to(device)

# cap = cv2.VideoCapture(0)
# while cap.isOpened():
# 	_, frame = cap.read()
# 	results = model(frame)
# 	detections = results.pandas().xyxy[0]

# 	# Extract the coordinates, class id, and confidence for each detection
# 	for i, detection in detections.iterrows():
# 		x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']]
# 		x1, y1, x2, y2 = [round(num) for num in [x1, y1, x2, y2]]

# 		class_id = detection['class']
# 		confidence = detection['confidence']
# 		#print(f'Detection {i}: class {class_id}, confidence {confidence}, bbox [{x1}, {y1}, {x2}, {y2}]')

# 		# Draw bounding box on input image
# 		cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 		# Put label text on bounding box
# 		label = f'{"Barcode"} {confidence:.2f}'
# 		label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
# 		cv2.rectangle(frame, (x1, y1), (x1 + label_size[0], y1 - label_size[1] - baseline), (0, 255, 0), cv2.FILLED)
# 		cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
		
# 		# Crop image to bounding box of first detected object
# 		cropped_img = frame[y1:y2, x1:x2]
# 		gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
# 		barcodes = pyzbar.decode(gray)
# 		for barcode in barcodes:
# 			data = barcode.data.decode("utf-8")

# 			# Put barcode data on bounding box
# 			data_size, baseline = cv2.getTextSize(data, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
# 			cv2.rectangle(frame, (x1, y2), (x1 + data_size[0], y2 + data_size[1]), (0, 255, 0), cv2.FILLED)
# 			cv2.putText(frame, data, (x1, y2 + data_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# 	# Display annotated image
# 	cv2.imshow('Result', frame)
# 	if cv2.waitKey(1) & ord('q') == 27:
# 		cv2.destroyAllWindows()
# 		break


import torch
import cv2
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

        # Crop and process barcode
        cropped_img = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        barcodes = pyzbar.decode(gray)
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
