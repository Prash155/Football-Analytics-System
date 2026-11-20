
import os
import cv2
import torch
import subprocess
import imageio_ffmpeg
import numpy as np
import streamlit as st

from collections import defaultdict, deque
from pathlib import Path
from ultralytics import YOLO
from sklearn.cluster import KMeans
from utils.heatmap import PlayerHeatmap


@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


def is_on_pitch(frame, x1, y1, x2, y2):

    height, width = frame.shape[:2]

    foot_x = max(0, min(width - 1, (x1 + x2) // 2))
    foot_y = max(0, min(height - 1, y2 - 4))

    radius = 12

    left = max(0, foot_x - radius)
    right = min(width, foot_x + radius + 1)
    top = max(0, foot_y - radius)
    bottom = min(height, foot_y + radius + 1)

    region = frame[top:bottom, left:right]

    if region.size == 0:
        return False

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(
        hsv,
        np.array([30, 35, 35]),
        np.array([95, 255, 255])
    )

    green_ratio = np.count_nonzero(green_mask) / green_mask.size

    return green_ratio >= 0.25


def process_video(video_path, output_folder="outputs", progress_callback=None):

    cv2.setNumThreads(1)
    torch.set_num_threads(4)

    os.makedirs(output_folder, exist_ok=True)

    model = load_model()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 25

    output_path = os.path.join(output_folder, "annotated_match.mp4")
    h264_path = os.path.join(output_folder, "annotated_match_h264.mp4")

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not create the output video.")

    heatmap = PlayerHeatmap(width, height)

    player_teams = defaultdict(lambda: deque(maxlen=20))
    player_samples = defaultdict(list)

    team_model = None
    training_frames = 90
    frame_stride = 2
    frame_count = 0
    processed_count = 0

    tracked_players = set()
    team_counts = {0: set(), 1: set()}

    last_annotated_frame = None

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            if (frame_count - 1) % frame_stride != 0:

                if last_annotated_frame is not None:
                    writer.write(last_annotated_frame)

                if progress_callback and total_frames > 0:
                    progress_callback(
                        min(frame_count / total_frames, 1.0)
                    )

                continue

            processed_count += 1

            results = model.track(
                frame,
                persist=True,
                classes=[0],
                tracker="bytetrack.yaml",
                imgsz=512,
                conf=0.25,
                verbose=False
            )

            players = []

            for box in results[0].boxes:

                if box.id is None:
                    continue

                player_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                box_height = y2 - y1
                box_width = x2 - x1

                if box_height <= 0 or box_width <= 0:
                    continue

                if not is_on_pitch(frame, x1, y1, x2, y2):
                    continue

                top = max(0, y1 + int(box_height * 0.18))
                bottom = min(height, y1 + int(box_height * 0.45))
                left = max(0, x1 + int(box_width * 0.25))
                right = min(width, x2 - int(box_width * 0.25))

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

                players.append(
                    (player_id, (x1, y1, x2, y2), features)
                )

                tracked_players.add(player_id)

                if team_model is None and processed_count <= training_frames:

                    if len(player_samples[player_id]) < 5:
                        player_samples[player_id].append(features)

            if team_model is None and processed_count >= training_frames:

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

            for player_id, (x1, y1, x2, y2), features in players:

                label = "Analyzing"
                color = (0, 255, 255)

                if team_model is not None:

                    team = int(team_model.predict([features])[0])
                    player_teams[player_id].append(team)

                    votes = player_teams[player_id]

                    team_1_votes = votes.count(0)
                    team_2_votes = votes.count(1)

                    if max(team_1_votes, team_2_votes) >= 12:

                        stable_team = (
                            0 if team_1_votes > team_2_votes else 1
                        )

                        label = f"Team {stable_team + 1}"

                        color = (
                            (0, 0, 255)
                            if stable_team == 0
                            else (255, 0, 0)
                        )

                        team_counts[stable_team].add(player_id)

                        if processed_count % 5 == 0:

                            center_x = (x1 + x2) // 2
                            foot_y = y2

                            heatmap.add_position(
                                player_id,
                                stable_team,
                                center_x,
                                foot_y
                            )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"{label} | ID: {player_id}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

            last_annotated_frame = frame.copy()
            writer.write(last_annotated_frame)

            if progress_callback and total_frames > 0:

                progress_callback(
                    min(frame_count / total_frames, 1.0)
                )

    finally:

        cap.release()
        writer.release()

    if frame_count == 0:
        raise ValueError("No frames could be processed.")

    for old_file in Path(output_folder).glob("*_heatmap.jpg"):
        old_file.unlink()

    heatmap.save_heatmaps(output_folder)

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    conversion = subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-i", output_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            h264_path
        ],
        capture_output=True,
        text=True
    )

    if conversion.returncode != 0:

        raise RuntimeError(
            "Video conversion failed: " + conversion.stderr[-2000:]
        )

    return {
        "output_video": h264_path,
        "total_frames": frame_count,
        "inference_frames": processed_count,
        "unique_players": len(tracked_players),
        "team_1_players": len(team_counts[0]),
        "team_2_players": len(team_counts[1]),
        "team_model_ready": team_model is not None
    }
