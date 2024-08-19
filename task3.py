import torch
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
from datetime import datetime

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# Initialize DeepSORT
deepsort = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2)


def create_output_folders(base_folder):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(base_folder, f"tracked_output_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def detect_and_track_person(video_path, output_base_folder):
    output_folder = create_output_folders(output_base_folder)
    persons_folder = os.path.join(output_folder, "persons")
    os.makedirs(persons_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(output_folder, 'output.mp4'), fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run YOLOv5 detection
        results = model(frame)

        # Extract person detections (class_id == 0 for persons in COCO dataset)
        detections = []
        for *xyxy, conf, cls in results.xyxy[0]:
            if int(cls) == 0:  # Only process persons
                x1, y1, x2, y2 = map(int, xyxy)
                width = x2 - x1
                height = y2 - y1
                detections.append(([x1, y1, width, height], conf.item()))

        # Update DeepSORT tracker
        tracks = deepsort.update_tracks(detections, frame=frame)

        # Draw tracking info on the frame and save individual frames
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_ltrb()
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Create folder for this person if it doesn't exist
            person_folder = os.path.join(persons_folder, f"person_{track_id}")
            os.makedirs(person_folder, exist_ok=True)

            # Save the frame for this person
            person_frame = frame[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(person_folder, f"frame_{frame_count}.jpg"), person_frame)

        # Write the frame to output video
        out.write(frame)

        # Display the frame (optional, you can comment this out for faster processing)
        cv2.imshow('Person Detection and Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Path to your video file
video_path = r"C:\Users\tragu\OneDrive\Pictures\Saved Pictures\WhatsApp Video 2024-08-16 at 17.36.20_b8d2efc4.mp4"

# Base folder for output
output_base_folder = r"C:\Users\tragu\OneDrive\Pictures\Saved Pictures\123"

# Run the detection and tracking function
detect_and_track_person(video_path, output_base_folder)