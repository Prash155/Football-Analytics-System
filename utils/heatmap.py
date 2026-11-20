
import cv2
import numpy as np
import os


class PlayerHeatmap:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.player_positions = {}
        self.team_positions = {
            0: [],
            1: []
        }

    def add_position(self, player_id, team, x, y):

        if player_id not in self.player_positions:
            self.player_positions[player_id] = []

        self.player_positions[player_id].append((x, y))

        if team in self.team_positions:
            self.team_positions[team].append((x, y))

    def generate_heatmap(self, positions):

        density = np.zeros(
            (self.height, self.width),
            dtype=np.float32
        )

        for x, y in positions:

            if 0 <= x < self.width and 0 <= y < self.height:

                cv2.circle(
                    density,
                    (int(x), int(y)),
                    30,
                    1.0,
                    -1
                )

        density = cv2.GaussianBlur(
            density,
            (0, 0),
            sigmaX=18,
            sigmaY=18
        )

        background = np.zeros(
            (self.height, self.width, 3),
            dtype=np.uint8
        )

        background[:] = (35, 75, 35)

        if np.max(density) == 0:
            return background

        normalized = cv2.normalize(
            density,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        colored = cv2.applyColorMap(
            normalized,
            cv2.COLORMAP_JET
        )

        mask = normalized > 5

        background[mask] = colored[mask]

        return background

    def save_heatmaps(self, output_folder="outputs"):

        os.makedirs(output_folder, exist_ok=True)

        for player_id, positions in self.player_positions.items():

            if len(positions) < 3:
                continue

            heatmap = self.generate_heatmap(positions)

            cv2.imwrite(
                os.path.join(
                    output_folder,
                    f"player_{player_id}_heatmap.jpg"
                ),
                heatmap
            )

        for team, positions in self.team_positions.items():

            heatmap = self.generate_heatmap(positions)

            cv2.imwrite(
                os.path.join(
                    output_folder,
                    f"team_{team + 1}_heatmap.jpg"
                ),
                heatmap
            )

        print("Player and team heatmaps saved successfully!")
