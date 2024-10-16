import os

import cv2
from ultralytics import YOLO

# Load the YOLOv8 model trained for pose detection (e.g., face, hands, body landmarks)
print("Loading Model...", end="")
model = YOLO("./models/yolo11m-pose.pt")
print("Done")

# Define the output directory and ensure it exists
output_dir = './output'  # Use relative path for the output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video file (you can also use 0 for the default webcam)
file_name = "test.mp4"
cap = cv2.VideoCapture(f"./data/{file_name}")

# Get video dimensions (width and height)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up VideoWriter to save the output video
all_output_files = os.listdir(output_dir)
past_output_files = [*filter(lambda file: file_name.split(".")[0] in file, all_output_files)]

output_path = f'{output_dir}/test-output-{len(past_output_files)+1}.mp4'  # Use the correct relative path
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
        break

    # Run the YOLO model on the frame for pose detection (face, hands, body)
    results = model.predict(source=frame, conf=0.3, max_det=15, verbose=True)

    # Annotate the frame with detected landmarks, bounding boxes, and labels
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video file
    out.write(annotated_frame)

    # Optionally display the video with annotations in a resizable window
    cv2.namedWindow('YOLOv11 Pose Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('YOLOv11 Pose Detection', width, height)
    cv2.imshow('YOLOv11 Pose Detection', annotated_frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture and the video writer, and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved to {output_path}")