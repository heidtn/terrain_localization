import numpy as np
from scipy.spatial.transform import Rotation as R

class Robot:
    def __init__(self, position: np.ndarray, orientation: np.ndarray, terrain: np.ndarray):
        self.position = np.array(position)
        self.orientation = R.from_euler("ZYX", orientation).as_matrix()
        self.terrain = terrain

    def project_to_terrain(self, start: np.ndarray, direction: np.ndarray):
        ray = np.array(start)
        while True:
            ray += direction
            if ray[0] > self.terrain.shape[0] or ray[0] < 0:
                return None
            if ray[1] > self.terrain.shape[1] or ray[1] < 0:
                return None
            if self.terrain[int(ray[0] + 0.5), int(ray[1] + 0.5)] > ray[2]:
                return np.linalg.norm(ray - start)

    def get_robot_view(self):
        # we're assuming flat ground for now
        x_pos = int(np.round(self.position[0]))
        y_pos = int(np.round(self.position[1]))
        current_altitude = self.terrain[x_pos, y_pos]

        