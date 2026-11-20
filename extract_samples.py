
import cv2
import os
import numpy as np

VIDEO_PATH = "videos/test 2.mp4"
OUTPUT_FOLDER = "sample_data"
SAMPLE_COUNT = 30

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise ValueError("Could not open the sample video.")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if total_frames < SAMPLE_COUNT:
    raise ValueError("Video does not contain enough frames.")

frame_indices = np.linspace(
    0, total_frames - 1, SAMPLE_COUNT, dtype=int
)

saved_count = 0

for index, frame_number in enumerate(frame_indices, start=1):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    ret, frame = cap.read()

    if not ret:
        print(f"Could not read frame {frame_number}")
        continue

    filename = os.path.join(
        OUTPUT_FOLDER,
        f"sample_{index:02d}.jpg"
    )

    cv2.imwrite(filename, frame)
    saved_count += 1

cap.release()

print(f"Dataset extraction completed!")
print(f"Total sample images saved: {saved_count}")
print(f"Location: {OUTPUT_FOLDER}")
