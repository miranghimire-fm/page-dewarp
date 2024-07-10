import numpy as np
from cv2 import solvePnP

from .options import K

__all__ = ["get_default_params"]

# TODO refactor this to be a class for both the flattening and parsing?
def get_default_params(corners, ycoords, xcoords):
    page_width, page_height = [np.linalg.norm(corners[i] - corners[0]) for i in (1, -1)]
    cubic_slopes = [0.0, 0.0]  # initial guess for the cubic has no slope
    # object points of flat page in 3D coordinates
    corners_object3d = np.array(
        [
            [0, 0, 0],
            [page_width, 0, 0],
            [page_width, page_height, 0],
            [0, page_height, 0],
        ]
    )
    # estimate rotation and translation from four 2D-to-3D point correspondences
    _, rvec, tvec = solvePnP(corners_object3d, corners, K(), np.zeros(5))
    span_counts = [*map(len, xcoords)]

    # print(f"Rvec: {rvec}, Length:{len(rvec)}")
    # print(f"Tvec: {tvec}, Length: {len(tvec)}")
    # print(f"f:Cubic Slopes: {cubic_slopes}")
    # print(f"Y coords: {ycoords}, Length: {len(ycoords)}")
    # print(f"X coords: {xcoords}, Length: {xcoords}")
    params = np.hstack(
        (
            np.array(rvec).flatten(),
            np.array(tvec).flatten(),
            np.array(cubic_slopes).flatten(),
            ycoords.flatten(),
        )
        + tuple(xcoords)
    )
    # print(f"Initial Default: {params}")
    return (page_width, page_height), span_counts, params
