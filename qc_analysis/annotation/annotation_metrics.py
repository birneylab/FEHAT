import numpy as np

from typing import List, Tuple, Iterable

import matplotlib.pyplot as plt

from shapely import geometry


def point_inside_polygon_q(polygon: List[List[float]], point: Iterable[float]) -> bool:
    """
    Check whether a point lies inside a polygon.
    """
    if len(point) != 2:
        raise ValueError(f"Point should have shape (2, ) or 2 but has shape {len(point)}.")
    
    point = geometry.Point(point[0], point[1])
    polygon = geometry.Polygon(polygon)
        
    return polygon.contains(point)

def count_metric(polygon: Iterable[Iterable[float]], roi_pixels: Iterable[Iterable[float]]) -> float:
    n_pixels = len(roi_pixels)
    m = sum(1 for pixel in roi_pixels if point_inside_polygon_q(polygon, pixel)) / n_pixels
    return (n_pixels - geometry.Polygon(polygon).area) / n_pixels, m

def com_metric(polygon: np.ndarray, roi_pixel_positions: np.ndarray) -> float:
    com_polygon = np.array(np.mean(polygon[:, 0]), np.mean(polygon[:, 1]))
    com_region = np.array(np.mean(roi_pixel_positions[:, 0]), np.mean(roi_pixel_positions[:, 1]))
    return np.linalg.norm(com_polygon - com_region)