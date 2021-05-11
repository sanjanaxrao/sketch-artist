import numpy as np
import cv2


def thinning(ccs):
    lines = []

    for cc in ccs:
        new_img = np.zeros(cc.shape)
        gray = np.copy(cc)
        (contours, _) = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # biggest area
        target = max(contours, key=lambda x: cv2.contourArea(x))
        
        x = target[:, :, 0].flatten()
        y = target[:, :, 1].flatten()
        poly = np.poly1d(np.polyfit(x, y, 5))

        draw_x = np.linspace(min(x), max(x), int((max(x) - min(x)) / 5))
        draw_y = np.polyval(poly, draw_x)
        draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)
        cv2.polylines(new_img, [draw_points], False, 255, thickness=1)

        lines.append(new_img)

    return lines