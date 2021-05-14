import numpy as np
import cv2
import time
import os
from custom_settings import user_control_frame_rate


def draw_animation(lines, shading_img, shading_lines):
    total_img = np.uint8(np.zeros(shading_img.shape))
    width, height = shading_img.shape
    out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                            user_control_frame_rate, (height, width), False)
    for line in lines:
        # add line to image
        total_img = cv2.add(np.uint8(line), total_img)
        # add to video
        to_draw = np.where(total_img == 0, 255, 0)
        out.write(np.uint8(to_draw))

    for shading_line in shading_lines:
        # add line to image
        total_img = cv2.add(np.uint8(shading_line), total_img)
        # add to video
        to_draw = np.where(total_img == 0, 255, 0)
        out.write(np.uint8(to_draw))

    out.release()


def composite_image(lines, shading):

    total_img = np.uint8(np.zeros(shading.shape))

    for line in lines:
        total_img = cv2.add(total_img, np.uint8(line))

    total_img = cv2.add(total_img, np.uint8(shading))
    total_img = np.where(total_img == 0, 255, 0)

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


def draw_lines(lines, shading_img, shading_lines, filepath, save_img=True, animate=True):
    if animate:
        print('animating output ' + str(filepath))
        final_img = draw_animation(lines, shading_img, shading_lines)
    else:
        print('drawing output ' + str(filepath))
        final_img = draw_still(lines, shading_img)

    if save_img:
        print('saving output ' + str(filepath))
        save_image(final_img, filepath)