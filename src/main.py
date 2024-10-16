import cv2
from ultralytics import YOLO
import os

# Load the YOLOv8 model trained for pose detection (e.g., face, hands, body landmarks)
print("Loading Model...", end="")
model = YOLO("./models/yolo11m-pose.pt")
print("Done")

# Define the output directory and ensure it exists
output_dir = './output'  # Use relative path for the output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video file (you can also use 0 for the default webcam)
cap = cv2.VideoCapture("./data/test.mp4")

# Get video dimensions (width and height)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up VideoWriter to save the output video
output_path = f'{output_dir}/test-output-2.mp4'  # Use the correct relative path
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 format
fps = int(cap.get(cv2.CAP_PROP_FPS)) 
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Check if the video is successfully opened
if not cap.isOpened():
    print("Error: Could not open video")
    exit()

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from video")
        break

    # Run the YOLO model on the frame for pose detection (face, hands, body)
    results = model.predict(source=frame, conf=0.25, max_det=10, verbose=True)

    # Annotate the frame with detected landmarks, bounding boxes, and labels
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video file
    out.write(annotated_frame)

    # Optionally display the video with annotations in a resizable window
    cv2.namedWindow('YOLOv11 Pose Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('YOLOv11 Pose Detection', width, height)
    cv2.imshow('YOLOv11 Pose Detection', annotated_frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and the video writer, and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved to {output_path}")