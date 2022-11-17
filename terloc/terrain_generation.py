import numpy as np
import argparse
from PIL import Image


class Terrain:
    terrain: np.ndarray
    resolution: float

    def __init__(self, resolution: float, terrain: np.ndarray):
        self.resolution = resolution
        self.terrain = terrain

    def save_as_image(self, filename: str):
        data = Image.fromarray(self.terrain)
        data.save(filename)


    def sample_terrain(
        self,
        center: np.ndarray,
        size: tuple[int, int],
        orientation: float,
        resolution,
    ):
        sample = np.zeros((size[0], size[1]))
        x_width = size[0]*resolution
        y_width = size[1]*resolution
        for x in range(size[0]):
            for y in range(size[1]):
                x_coord_local = (x*resolution) - (x_width / 2.0) 
                y_coord_local = (y*resolution) - (y_width / 2.0) 
                coords = np.array([x_coord_local, y_coord_local])
                rot_mat = get_R_2D(orientation)
                rotated_coords = np.dot(rot_mat, coords)
                transformed_coords = rotated_coords + center
                height = self.sample_point(transformed_coords)
                sample[x, y] = height

        return sample

    def sample_point(self, coordinates: np.ndarray) -> float:
        # for now, direct sample, eventually we can do a weight of neighbors
        coordinates /= self.resolution
        coordinates = np.round(coordinates).astype(np.int64)
        return self.terrain[coordinates[0], coordinates[1]]

def show_image(terrain: np.ndarray):
    toshow = (terrain / np.max(terrain)) * 255
    im = Image.fromarray(terrain)
    im.show()

def get_R_2D(theta: float):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def generate_perlin_noise_2d(shape, res) -> np.ndarray:
    """pulled from here: https://github.com/pvigier/perlin-numpy/blob/master/perlin_numpy/perlin2d.py"""

    def f(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def get_terrain_gradient(terrain: np.ndarray) -> np.ndarray:
    grads = np.gradient(terrain)
    gradient = np.zeros((terrain.shape[0], terrain.shape[1], 2))
    gradient[:, :, 0] = grads[0]
    gradient[:, :, 1] = grads[1]
    return gradient


def create_terrain(
    width: int,
    length: int,
    min_height: float,
    max_height: float,
    res: tuple[float, float],
    resolution: float,
) -> Terrain:
    terrain_array = generate_perlin_noise_2d((width, length), res)
    terrain_array = (terrain_array / np.max(terrain_array)) * max_height + min_height
    terrain = Terrain(resolution, terrain_array)

    return terrain


def main():
    np.random.seed(12345)
    terrain = create_terrain(512, 512, 0, 30, (8, 8), 1.0)
    section = terrain.sample_terrain(np.array([256, 256]), (128, 128), np.deg2rad(60), 1.0)
    show_image(terrain.terrain)
    show_image(section)


if __name__ == "__main__":
    main()
