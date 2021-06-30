import numpy as np
from typing import Tuple
from scipy import ndimage
import cv2 as cv


def gaussian_blur(img: np.ndarray, kernel: Tuple[int, int], sigma: float) -> np.ndarray:
    return cv.GaussianBlur(img, kernel, sigma)


def magnitude_and_direction(img: np.ndarray, kernel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    image_height, img_width = img.shape
    pad_y = (kernel.shape[0] - 1) // 2
    pad_x = (kernel.shape[1] - 1) // 2
    img_padded = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x)), 'reflect')

    edges_y = ndimage.convolve(img_padded, kernel)[pad_y:image_height + pad_y, pad_x:img_width + pad_x]
    edges_x = ndimage.convolve(img_padded, kernel.T)[pad_y:image_height + pad_y, pad_x:img_width + pad_x]

    magnitude = np.hypot(edges_y, edges_x)
    direction = np.arctan2(edges_x, edges_y)
    direction[direction < 0] += 2 * np.pi

    return magnitude, direction


def edge_thinning(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    mask = np.full(magnitude.shape, False)
    direction[direction > np.pi] -= np.pi

    up = magnitude.copy()
    down = magnitude.copy()
    right = magnitude.copy()
    left = magnitude.copy()
    up_left = magnitude.copy()
    up_right = magnitude.copy()
    down_left = magnitude.copy()
    down_right = magnitude.copy()
    down = np.roll(down, 1, axis=0)
    right = np.roll(right, 1, axis=1)
    up = np.roll(up, -1, axis=0)
    left = np.roll(left, -1, axis=1)
    down_right = np.roll(down_right, (1, 1), axis=(0, 1))
    down_left = np.roll(down_left, (1, -1), axis=(0, 1))
    up_right = np.roll(up_right, (-1, 1), axis=(0, 1))
    up_left = np.roll(up_left, (-1, -1), axis=(0, 1))

    # 0/180
    mask[(((direction <= np.pi / 8) & (direction >= 0)) | ((direction <= np.pi) & (direction > 7 * np.pi / 8))) & (magnitude > right) & (magnitude > left)] = True

    # 45
    mask[np.logical_and(np.logical_and(np.logical_and(direction > np.pi / 8, direction <= 3 * np.pi / 8), magnitude > up_left), magnitude > down_right)] = True
    #
    # 90
    mask[np.logical_and(np.logical_and(np.logical_and(direction > 3 * np.pi / 8, direction <= 5 * np.pi / 8), magnitude > up), magnitude > down)] = True

    # 135
    mask[np.logical_and(np.logical_and(np.logical_and(direction > 5 * np.pi / 8, direction <= 7 * np.pi / 8), magnitude > up_right), magnitude > down_left)] = True

    return mask


def edge_tracking(magnitude: np.ndarray, mask: np.ndarray, low_threshold: float, high_threshold: float) -> np.ndarray:
    magnitude_ = magnitude.copy()
    magnitude_[np.logical_not(mask)] = 0

    strong_mask = magnitude_ > high_threshold
    weak_mask = (magnitude_ > low_threshold) & (magnitude_ < high_threshold)

    new_strong = []

    for y in range(magnitude_.shape[0]):
        for x in range(magnitude_.shape[1]):
            if strong_mask[y][x]:
                new_strong.append((y, x))
                start = -1 if y > 0 else 0
                end = 2 if y < magnitude_.shape[0] - 1 else 1
                for j in range(start, end):
                    start1 = -1 if x > 0 else 0
                    end1 = 2 if x < magnitude_.shape[1] - 1 else 1
                    for k in range(start1, end1):
                        if weak_mask[y + j][x + k]:
                            new_strong.append((y + j, x + k))
                            weak_mask[y + j][x + k] = False
                            strong_mask[y + j][x + k] = True

    while len(new_strong):
        y, x = new_strong.pop()

        strong_mask[y][x] = True

        start = -1 if y > 0 else 0
        end = 2 if y < magnitude_.shape[0] - 1 else 1
        for j in range(start, end):
            start1 = -1 if x > 0 else 0
            end1 = 2 if x < magnitude_.shape[1] - 1 else 1
            for k in range(start1, end1):
                if weak_mask[y + j][x + k]:
                    new_strong.append((y + j, x + k))
                    weak_mask[y + j][x + k] = False

    strong_mask[np.logical_not(mask)] = False
    return strong_mask