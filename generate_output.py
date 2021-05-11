import numpy as np
import cv2
import time
import os


def pts_to_spline(img_size, pts):
    spline = np.zeros(img_size)
    for i in range(len(pts) - 1):
        cv2.line(spline, tuple(pts[i]), tuple(pts[i+1]), 255, thickness=1)
    return np.uint8(spline)


def composite_image(lines, shading):

    total_img = np.uint8(np.zeros(shading.shape))

    for line in lines:
        total_img = cv2.add(total_img, np.uint8(line))

    total_img = cv2.add(total_img, np.uint8(shading))
    total_img = np.where(total_img == 0, 255, 0)

    global curr_image
    curr_image = np.ones(total_img.shape) * 255

    return total_img


def draw_still(lines, shading):
    total_img = composite_image(lines, shading)

    cv2.imshow('final result', np.uint8(total_img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return total_img


def save_image(img, filepath):
    filename = filepath.split('/')[1]
    filename = filename.split('.')[0]
    time_str = time.strftime("%Y%m%d-%H%M%S")
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'result_images', filename + '_' + time_str)
    if not cv2.imwrite(dir_path, img):
        raise Exception("Could not write image")


def draw_lines(lines, shading, filepath, save_img=True):

    final_img = draw_still(lines, shading)

    if save_img:
        save_image(final_img, filepath)