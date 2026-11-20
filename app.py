
import cv2
import numpy as np
import torch
from collections import defaultdict, deque
from ultralytics import YOLO
from sklearn.cluster import KMeans
from utils.heatmap import PlayerHeatmap

cv2.setNumThreads(1)
torch.set_num_threads(4)

model = YOLO("yolo11n.pt")

video_path = "videos/test 2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video!")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

heatmap = PlayerHeatmap(width, height)

player_teams = defaultdict(lambda: deque(maxlen=20))
player_samples = defaultdict(list)

cv2.namedWindow("Football Analytics System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Football Analytics System", 960, 540)

frame_count = 0
team_model = None
training_frames = 90

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        tracker="bytetrack.yaml",
        imgsz=640,
        conf=0.25,
        verbose=False
    )

    players = []

    for box in results[0].boxes:
        if box.id is None:
            continue

        player_id = int(box.id[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        player_height = y2 - y1
        player_width = x2 - x1

        if player_height <= 0 or player_width <= 0:
            continue

        top = max(0, y1 + int(player_height * 0.18))
        bottom = min(frame.shape[0], y1 + int(player_height * 0.45))
        left = max(0, x1 + int(player_width * 0.25))
        right = min(frame.shape[1], x2 - int(player_width * 0.25))

        jersey = frame[top:bottom, left:right]

        if jersey.size == 0:
            continue

        hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3)

        valid_pixels = pixels[
            (pixels[:, 1] > 40) &
            (pixels[:, 2] > 50)
        ]

        if len(valid_pixels) < 10:
            continue

        features = np.median(valid_pixels, axis=0)

        players.append((player_id, (x1, y1, x2, y2), features))

        if team_model is None and frame_count <= training_frames:
            if len(player_samples[player_id]) < 5:
                player_samples[player_id].append(features)

    if team_model is None and frame_count >= training_frames:
        all_samples = [
            sample
            for samples in player_samples.values()
            for sample in samples
        ]

        if len(all_samples) >= 6:
            team_model = KMeans(
                n_clusters=2,
                random_state=42,
                n_init=10
            )

            team_model.fit(np.array(all_samples))

            print("Team identification model ready!")

    for player_id, (x1, y1, x2, y2), features in players:

        if team_model is not None:
            team = int(team_model.predict([features])[0])

            player_teams[player_id].append(team)

            votes = player_teams[player_id]

            team_1_votes = votes.count(0)
            team_2_votes = votes.count(1)

            if max(team_1_votes, team_2_votes) >= 12:
                stable_team = 0 if team_1_votes > team_2_votes else 1

                label = f"Team {stable_team + 1}"

                color = (
                    (0, 0, 255)
                    if stable_team == 0
                    else (255, 0, 0)
                )

                if frame_count % 5 == 0:
                    center_x = (x1 + x2) // 2
                    foot_y = y2

                    heatmap.add_position(
                        player_id,
                        stable_team,
                        center_x,
                        foot_y
                    )

            else:
                label = "Analyzing"
                color = (0, 255, 255)

        else:
            label = "Analyzing Teams"
            color = (0, 255, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            frame,
            f"{label} | ID: {player_id}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    display_frame = cv2.resize(
        frame,
        (960, 540),
        interpolation=cv2.INTER_AREA
    )

    cv2.imshow("Football Analytics System", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

heatmap.save_heatmaps("outputs")

print("Football analytics completed!")
print("Player and team heatmaps saved in outputs folder!")
