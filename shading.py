import cv2
import numpy as np
import math
import random
from scipy.interpolate import splprep, splev
from custom_settings import user_control_num_lines, user_control_line_length, user_control_amplitude, user_control_curviness, user_control_num_oriented


def get_thesholds(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    num_pixels = img.shape[0] * img.shape[1]
    num_shades = 3

    # NUMBER OF SHADES
    num_in_bucket = num_pixels * .6 / (num_shades-1)
    thresholds = []

    curr_count = 0
    for i in range(len(hist)):
        curr_count += hist[i]

        if curr_count >= num_in_bucket:
            thresholds.append(i)
            curr_count = 0

            if len(thresholds) == num_shades - 1:
                break

    return thresholds


def threshold_imgs(img):
    thresholds = get_thesholds(img)

    imgs = []
    thresholds_as_fractions = [0]

    for i in range(len(thresholds)):
        curr_thresh = thresholds[i]

        prev_thresh = int(thresholds_as_fractions[i] * 255.)
        thresh_img = cv2.inRange(img, prev_thresh, curr_thresh)

        imgs.append(thresh_img)
        thresholds_as_fractions.append(curr_thresh / 255.)

    return imgs, thresholds_as_fractions[1:]


def get_ccs(thresholded, thresholds):
    ccs = []
    new_thresholds = []
    areas = []

    for i, im in enumerate(thresholded):

        # Area and bounding box
        output = cv2.connectedComponentsWithStats(im.astype(np.uint8))

        num_labels = output[0]
        labels = output[1]
        stats = output[2]

        for label in range(1, num_labels):
            if stats[label][cv2.CC_STAT_AREA] < 50:
                continue

            pts = np.where(labels == label)
            bounding_box = stats[label][cv2.CC_STAT_WIDTH] * stats[label][cv2.CC_STAT_HEIGHT]

            if len(pts[0]) / bounding_box < .2:
                continue

            cc = np.array(labels, dtype=np.uint8)
            cc[labels == label] = 255
            cc[labels != label] = 0

            ccs.append(cc)
            new_thresholds.append(thresholds[i])
            areas.append(stats[label][cv2.CC_STAT_AREA])

    return ccs, new_thresholds, areas


def get_area_orientation(pt1, pt2, gray_img):
    x_min = min(pt1[0], pt2[0])
    x_max = max(pt1[0], pt2[0])
    y_min = min(pt1[1], pt2[1])
    y_max = max(pt1[1], pt2[1])

    area_gradient, area_norm = get_orientation(gray_img, x_min, y_min, x_max - x_min, y_max - y_min)

    return area_gradient, area_norm


def conditions(prev, curr, min_dist, max_dist, gray_img, ignore_orientation):
    # conditon 1: not too far apart
    max_dist = min_dist <= math.dist(prev, curr) <= max_dist
    if not max_dist:
        return False

    # condition 2: not same point
    not_same = prev != curr

    # condition 3: correct angle
    o_range = math.pi / 6.

    line_orientation = math.atan2(curr[1]-prev[1], curr[0]-prev[0])
    area_gradient, area_norm = get_area_orientation(prev, curr, gray_img)

    orientation = ignore_orientation or area_gradient - o_range <= line_orientation <= area_gradient + o_range

    return max_dist and not_same and orientation


def create_point_set(blob, threshold, area, gray_img):
    # getting all points that are white, make tuples
    y, x = np.where(blob == 255)
    all_pairs = [(x[i], y[i]) for i in range(len(x))]

    random.shuffle(all_pairs)

    error_image = np.zeros(blob.shape) # where lines are drawn
    num_shaded_pixels = 0

    num_to_be_shaded = user_control_num_lines * int(len(x) * (1. - threshold))

    vol = (max(x) - min(x)) * (max(y) - min(y))

    max_dist = user_control_line_length * int(min(max(x) - min(x), max(y) - min(y)) * 1.2 * (area/vol))

    min_dist = int(max_dist / 2.)

    pts = [random.choice(all_pairs)]

    # inf loop shit
    count = 0
    # while loop until error image is satisfied
    while num_shaded_pixels < num_to_be_shaded:
        prev = pts[len(pts) - 1]
        curr = random.choice(all_pairs)

        # first point of line
        if len(pts) % 2 == 1:
            iter_count = 0
            ignore_orientation = False
            if len(pts) % user_control_num_oriented == 0:
                ignore_orientation = True
            while not conditions(prev, curr, min_dist, max_dist, gray_img, ignore_orientation):
                curr = random.choice(all_pairs)
                iter_count += 1

                # edge case handling
                if iter_count > 50:
                    ignore_orientation = True

        pts.append(curr)

        # second point of line: if we have a new line , draw it
        if len(pts) % 2 == 0:
            # onto white image
            error_image = cv2.line(error_image, prev, curr, 255, 1)
            curr_y, curr_x = np.where(error_image == 255)
            prev_num_shaded = num_shaded_pixels
            num_shaded_pixels = len(curr_x)

            # edge case handling
            if prev_num_shaded == num_shaded_pixels:
                count += 1

            if count > 40:
                break
    return pts


def normalize_angle(angle):
    new_angle = angle
    while new_angle <= -180:
        new_angle += 360
    while new_angle > 180:
        new_angle -= 360
    return new_angle


def get_orientation(gray_img, x, y, w, h):
    blob = gray_img[y:y + h, x:x + w]

    if w <= 0 or h <= 0:
        return -1, -1

    sobelx = cv2.Sobel(blob, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(blob, cv2.CV_64F, 0, 1, ksize=5)

    x_avg = np.average(sobelx)
    y_avg = np.average(sobely)

    if x_avg == 0:
        return -1, -1

    gradient = math.atan(y_avg / x_avg)

    if math.isnan(gradient):
        return -1, -1

    normal = gradient + math.radians(90)
    normal = math.radians(normalize_angle(math.degrees(normal)))

    if math.isnan(normal):
        normal = -1

    return normal, gradient


def pts_to_spline(img_size, pts):
    spline = np.zeros(img_size)
    for i in range(len(pts) - 1):
        cv2.line(spline, (int(pts[i][0]), int(pts[i][1])), (int(pts[i+1][0]), int(pts[i+1][1])), 255, thickness=1)
    return np.uint8(spline)


def create_spline_pts(pts, blob_shape):
    if len(pts) == 0:
        return []

    pts = np.array(pts)

    try:
        tck, u = splprep(pts.T, u=None, s=0.0)
    except:
        return pts

    dist = 0
    for i in range(len(pts) - 1):
        dist += math.sqrt(((pts[i+1][0] - pts[i][0]) ** 2) + ((pts[i+1][1] - pts[i][1]) ** 2))

    if dist < 20:
        return pts

    u_new = np.linspace(u.min(), u.max(), int(dist))
    x_new, y_new = splev(u_new, tck, der=0)

    x_new = np.array(x_new).astype(int)
    y_new = np.array(y_new).astype(int)

    x_new = np.clip(x_new, 0, int(blob_shape[1] - 1))
    y_new = np.clip(y_new, 0, int(blob_shape[0] - 1))
    spline_pts = np.array(list((zip(x_new, y_new))))

    return spline_pts


def draw_stroke(pointA, pointB, area, gray_img):
    num_pts = 10
    min_side_step = 5
    side_step = user_control_amplitude

    if side_step > math.dist(pointA, pointB) and area < 1000:
        min_side_step = 2
        side_step = min(side_step, int(math.dist(pointA, pointB)) * 2)

    if (pointB[0] - pointA[0]) == 0:
        return [pointA, pointB]

    slope = (pointB[1] - pointA[1]) / (pointB[0] - pointA[0])
    b = pointA[1] - slope * pointA[0]
    mid_pt = ((pointA[0] + pointB[0]) / 2., (pointA[1] + pointB[1]) / 2.)

    grad, norm = get_area_orientation(pointA, pointB, gray_img)

    if grad < 0 or norm < 0:
        mid_pt_x = mid_pt[0] + random.randrange(-side_step, side_step)
        mid_pt_y = mid_pt[1] + random.randrange(-side_step, side_step)
        mid_pt = (int(mid_pt_x), int(mid_pt_y))
    else:
        x_multiplier = -1 if grad <= math.pi / 4 or grad > 3 * math.pi / 4 else 1
        y_multiplier = -1 if grad > math.pi / 2. else 1

        mid_pt_x = mid_pt[0] + x_multiplier * random.randrange(min_side_step, side_step) * math.cos(norm)
        mid_pt_y = mid_pt[1] + y_multiplier * random.randrange(min_side_step, side_step) * math.sin(norm)
        mid_pt = (int(mid_pt_x), int(mid_pt_y))

    shuffle = user_control_curviness
    mid_A = (int((pointA[0] + mid_pt[0]) / 2.) + random.randrange(-shuffle, shuffle),
             int((pointA[1] + mid_pt[1]) / 2.) + random.randrange(-shuffle, shuffle))
    mid_B = (int((mid_pt[0] + pointB[0]) / 2.) + random.randrange(-shuffle, shuffle),
             int((mid_pt[1] + pointB[1]) / 2.) + random.randrange(-shuffle, shuffle))

    return [pointA, mid_A, mid_pt, mid_B, pointB]


def perturb_points(pts, perturb_value=1):
    for i in range(0, len(pts), 3):
        new_x = pts[i][0] + random.randrange(-perturb_value, perturb_value)
        new_y = pts[i][1] + random.randrange(-perturb_value, perturb_value)
        pts[i] = (int(new_x), int(new_y))

    return pts


def create_lines(gray_img, blobs, thresholds, areas):
    imgs = []
    shape = gray_img.shape

    # for each blob create a new point gets
    for i, blob in enumerate(blobs):
        pts = create_point_set(blob, thresholds[i], areas[i], gray_img)

        img = np.uint8(np.zeros(shape))
        for j in range(0, len(pts) - 1, 2):
            to_draw = draw_stroke(pts[j], pts[j + 1], areas[i], gray_img)
            spline_points = create_spline_pts(to_draw, shape)
            perturbed_points = perturb_points(spline_points)

            img = cv2.add(img, pts_to_spline(shape, perturbed_points))
        imgs.append(np.uint8(img))

    return imgs


def shade(gray_img):
    thresh_imgs, thresholds = threshold_imgs(cv2.blur(gray_img, (10, 10)))
    ccs, thresholds, areas = get_ccs(thresh_imgs, thresholds)
    lines = create_lines(gray_img, ccs, thresholds, areas)

    total_img = np.uint8(np.zeros((gray_img.shape[0], gray_img.shape[1])))

    for l in lines:
        total_img = cv2.add(total_img, np.uint8(l))

    return total_img