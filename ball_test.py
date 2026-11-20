
import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

video_path = "videos/test 2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video!")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    if frame_count % 2 != 0:
        continue

    results = model(
        frame,
        classes=[32],
        conf=0.25,
        imgsz=640,
        max_det=1,
        verbose=False
    )

    annotated_frame = results[0].plot()
    display_frame = cv2.resize(annotated_frame, (960, 540))

    cv2.imshow("Football Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Ball detection test completed!")