
import os
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image

from utils.processor import process_video

st.set_page_config(
    page_title="Football Analytics System",
    page_icon="⚽",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

st.markdown("""
<style>
.stApp {
    background-color: #10141c;
    color: #ffffff;
}
[data-testid="stSidebar"] {
    background-color: #171e29;
}
div[data-testid="stMetric"] {
    background-color: #1c2635;
    padding: 18px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

results = st.session_state.analysis_results

st.title("⚽ Football Analytics System")
st.caption("Computer Vision Based Football Match Analysis")

st.sidebar.title("Football Analytics")
st.sidebar.caption("Computer Vision Match Analysis")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Video Analysis",
        "Team Analytics",
        "Heatmaps"
    ]
)

st.sidebar.markdown("---")
st.sidebar.success("System Ready")

if page == "Dashboard":

    st.header("Match Overview")

    if results:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Tracked Person IDs",
            results["unique_players"]
        )

        col2.metric(
            "Team 1 Assigned IDs",
            results["team_1_players"]
        )

        col3.metric(
            "Team 2 Assigned IDs",
            results["team_2_players"]
        )

        st.success("Football analysis completed successfully.")

        col1, col2 = st.columns(2)

        col1.metric(
            "Total Video Frames",
            results["total_frames"]
        )

        col2.metric(
            "Frames Analysed by YOLO",
            results.get("inference_frames", results["total_frames"])
        )

    else:

        col1, col2, col3 = st.columns(3)

        col1.metric("Player Detection", "YOLO")
        col2.metric("Player Tracking", "ByteTrack")
        col3.metric("Team Identification", "K-Means")

        st.info("Upload a football match video to begin.")

    st.markdown("---")

    st.header("System Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Player Analysis")
        st.write("Automatic player detection")
        st.write("Unique player tracking IDs")
        st.write("Bounding box visualization")

    with col2:
        st.markdown("### Match Analytics")
        st.write("Automatic team clustering")
        st.write("Team comparison")
        st.write("Individual and team heatmaps")

elif page == "Video Analysis":

    st.header("Football Video Analysis")

    uploaded_video = st.file_uploader(
        "Upload a football match video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:

        st.video(uploaded_video)

        if st.button(
            "Start Football Analysis",
            type="primary"
        ):

            progress_bar = st.progress(0)
            status = st.empty()
            temp_path = None

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(uploaded_video.name).suffix
                ) as temp_file:

                    temp_file.write(uploaded_video.getbuffer())
                    temp_path = temp_file.name

                status.info(
                    "Processing video. Please wait..."
                )

                new_results = process_video(
                    temp_path,
                    str(OUTPUT_DIR),
                    progress_callback=progress_bar.progress
                )

                st.session_state.analysis_results = new_results
                results = new_results

                progress_bar.progress(1.0)

                status.success(
                    "Football analysis completed successfully!"
                )

                st.rerun()

            except Exception as error:

                status.error(
                    f"Analysis failed: {error}"
                )

            finally:

                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

    results = st.session_state.analysis_results

    if results:

        st.markdown("---")
        st.subheader("Analysis Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Tracked Person IDs",
            results["unique_players"]
        )

        col2.metric(
            "Team 1 Assigned IDs",
            results["team_1_players"]
        )

        col3.metric(
            "Team 2 Assigned IDs",
            results["team_2_players"]
        )

        st.write(
            f"Total frames: {results['total_frames']}"
        )

        st.write(
            f"Frames analysed: {results.get('inference_frames', results['total_frames'])}"
        )

        if results["team_model_ready"]:
            st.success("Team identification model generated.")
        else:
            st.warning("Team identification model was not generated.")

        video_path = results["output_video"]

        if os.path.exists(video_path):

            st.markdown("---")
            st.subheader("Processed Match Video")

            st.video(video_path)

            with open(video_path, "rb") as video_file:

                st.download_button(
                    "Download Processed Match Video",
                    data=video_file,
                    file_name="football_analysis.mp4",
                    mime="video/mp4"
                )

elif page == "Team Analytics":

    st.header("Team Analytics")

    if results:

        team_1 = results["team_1_players"]
        team_2 = results["team_2_players"]
        total = results["unique_players"]

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Tracked IDs", total)
        col2.metric("Team 1 Assigned IDs", team_1)
        col3.metric("Team 2 Assigned IDs", team_2)

        st.markdown("---")

        st.subheader("Team Comparison")

        chart_data = pd.DataFrame({
            "Team": ["Team 1", "Team 2"],
            "Unique Assigned IDs": [team_1, team_2]
        })

        st.bar_chart(
            chart_data,
            x="Team",
            y="Unique Assigned IDs",
            color="#3985e8"
        )

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Team 1")
            st.metric("Unique Assigned IDs", team_1)
            st.caption("Identified using K-Means clustering.")

        with col2:
            st.subheader("Team 2")
            st.metric("Unique Assigned IDs", team_2)
            st.caption("Identified using K-Means clustering.")

        st.markdown("---")

        st.subheader("Identification Status")

        if results["team_model_ready"]:
            st.success("K-Means identification model generated successfully.")
        else:
            st.warning("Team identification model was not generated.")

        st.info(
            "These are unique tracking IDs assigned to each team during "
            "the video. They are not live player counts or confirmed team "
            "rosters. An ID may be assigned to different teams at different times."
        )

    else:

        st.info(
            "Upload and analyse a football video to view team analytics."
        )

elif page == "Heatmaps":

    st.header("Player and Team Heatmaps")

    heatmap_files = sorted(
        file for file in OUTPUT_DIR.glob("*_heatmap.jpg")
    )

    team_files = [
        file for file in heatmap_files
        if file.name.startswith("team_")
    ]

    player_files = [
        file for file in heatmap_files
        if file.name.startswith("player_")
    ]

    st.subheader("Team Heatmaps")

    if team_files:

        col1, col2 = st.columns(2)

        for index, file in enumerate(team_files):

            with (col1 if index % 2 == 0 else col2):

                st.markdown(
                    f"**{file.stem.replace('_', ' ').title()}**"
                )

                st.image(
                    Image.open(file),
                    width="stretch"
                )

    else:

        st.info("No team heatmaps available. Run an analysis first.")

    st.markdown("---")

    st.subheader("Individual Player Heatmaps")

    if player_files:

        player_ids = [
            file.stem.replace("player_", "").replace("_heatmap", "")
            for file in player_files
        ]

        selected_player = st.selectbox(
            "Select a player tracking ID",
            player_ids
        )

        selected_file = OUTPUT_DIR / f"player_{selected_player}_heatmap.jpg"

        if selected_file.exists():

            st.markdown(f"**Player ID: {selected_player}**")

            st.image(
                Image.open(selected_file),
                width="stretch"
            )

    else:

        st.info("No individual player heatmaps available.")

    st.caption(
        "Heatmaps represent detected positions in video-frame coordinates. "
        "They are not calibrated pitch-coordinate maps."
    )

st.markdown("---")
st.caption(
    "Football Analytics System | Computer Vision Based Match Analysis"
)