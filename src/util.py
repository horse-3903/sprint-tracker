import os
from pathlib import Path
import cv2
from ultralytics import YOLO
from tqdm import tqdm

class ModelNotFoundError(Exception):
    pass

class SprintTracker:
    def __init__(self, model: str, input_file: str, output_file: str):
        if model.split("/")[1] not in [m for m in os.listdir("models")]:
            raise ModelNotFoundError(f"Model {model} not found, please include the parent directory 'model/[model-name]' if you have not")

        print("Loading Model...", end="")
        self.model = YOLO(model)
        print("Done")

        if not os.path.exists(os.path.abspath(input_file)):
            raise FileNotFoundError(f"Input file {os.path.abspath(input_file)} not found")
        
        output_path = Path(output_file)

        if not os.path.exists(os.path.abspath(output_path.parent.absolute())):
            raise FileNotFoundError(f"Output directory {output_path.parent.absolute()} not found")
        if os.path.exists(os.path.abspath(output_file)):
            raise FileExistsError(f"Output file {os.path.abspath(output_file)} already exists")

        self.input_file = input_file
        self.output_file = output_file

    def crop_video(self):
        cap = cv2.VideoCapture(self.input_file)
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read video frame.")
            return None
        
        # Set the window to the same size as the video frame
        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select ROI", frame.shape[1], frame.shape[0])
        
        # Allow the user to select the ROI
        roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
        cap.release()
        
        return roi

    def track_sprint(self, conf: float = 0.5, max_det: int = 1, verbose: bool = False, save: bool = False):
        roi = self.crop_video()
        if roi is None:
            return

        x, y, w, h = roi
        cap = cv2.VideoCapture(self.input_file)
        width = w
        height = h

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if save:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS)) 
            out = cv2.VideoWriter(self.output_file, fourcc, fps, (width, height))
        
        if not cap.isOpened():
            print("Error: Could not open video")
            exit()

        for frame in tqdm(range(frame_count)):
            ret, frame = cap.read()

            if not ret:
                break

            # Crop the frame based on ROI
            cropped_frame = frame[y:y+h, x:x+w]

            results = self.model.predict(source=cropped_frame, conf=conf, max_det=max_det, verbose=verbose)

            annotated_frame = results[0].plot()

            if save:
                out.write(annotated_frame)

            cv2.namedWindow('YOLOv11 Pose Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('YOLOv11 Pose Detection', width, height)
            cv2.imshow('YOLOv11 Pose Detection', annotated_frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()

        if save:
            out.release()

        cv2.destroyAllWindows()

        print(f"Output video saved to {self.output_file}")
        
if __name__ == "__main__":
    st = SprintTracker(model="models/yolo11x-pose.pt", input_file="input/test-1.mp4", output_file="output/test-output-4.mp4")
    st.track_sprint(conf=0.1, max_det=12, verbose=False, save=True)