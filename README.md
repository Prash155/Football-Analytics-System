
# Football Analytics System Using Computer Vision

## Project Overview

The Football Analytics System is a computer vision-based application developed to analyse football match videos. It uses deep learning and machine learning techniques to detect players, track their movements, identify teams and generate movement heatmaps.

The system provides an interactive dashboard built using Streamlit, where users can upload football videos and view the analysis results.

## Features

- Automatic player detection using YOLO11.
- Player tracking using ByteTrack.
- Automatic team identification using K-Means clustering.
- Bounding box and player ID visualization.
- Football video processing and annotation.
- Individual player movement heatmaps.
- Team movement heatmaps.
- Interactive Streamlit dashboard.
- Processed video download.

## Technologies Used

- Python
- OpenCV
- YOLO11
- ByteTrack
- Scikit-learn
- K-Means Clustering
- NumPy
- Pandas
- Streamlit
- FFmpeg

## Project Structure

```text
Football_Analytics_System/
│
├── app.py
├── dashboard.py
├── requirements.txt
├── README.md
├── yolo11n.pt
│
├── models/
│
├── videos/
│
├── outputs/
│
└── utils/
    ├── processor.py
    └── heatmap.py
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the Project Folder

```bash
cd Football_Analytics_System
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run dashboard.py
```

## How to Use

1. Open the application in your browser.
2. Navigate to the Video Analysis section.
3. Upload a football match video.
4. Click Start Football Analysis.
5. Wait for the video processing to finish.
6. View the annotated video and download the output.
7. Open Team Analytics to compare team assignments.
8. Visit Heatmaps to view individual and team movement visualizations.

## Working Methodology

### Player Detection

YOLO11 is used to detect people in football match footage.

### Player Tracking

ByteTrack assigns tracking IDs to detected people and maintains their identities across video frames.

### Team Identification

K-Means clustering groups detected players based on jersey colour features extracted from HSV images.

### Heatmap Generation

Player positions are collected during video processing. OpenCV is used to generate heatmaps representing movement density in video-frame coordinates.

## Limitations

- Player detection may be affected by occlusion, camera angles and video quality.
- Team identification is based on jersey-colour clustering and may occasionally misclassify players.
- The system does not guarantee referee exclusion.
- Heatmaps use video-frame coordinates and are not calibrated football pitch maps.
- Processing time depends on video length and hardware capabilities.

## Future Scope

- Improved football pitch detection.
- More accurate player and referee classification.
- Camera-motion compensation.
- Tactical heatmaps using pitch coordinates.
- Football movement and performance analysis.
- Ball detection and trajectory tracking.

## Project Information

**Project Title:** Football Analytics System Using Computer Vision

**Domain:** Computer Vision and Deep Learning

**Framework:** Streamlit

**Primary Model:** YOLO11

**Tracking Algorithm:** ByteTrack

**Team Identification:** K-Means Clustering
## Live Deployment

The Football Analytics System is deployed using Streamlit Community Cloud.

Live Demo: https://football-analytics-system-92u83e6bv62scc4svotget.streamlit.app/