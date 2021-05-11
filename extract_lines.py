import numpy as np
import cv2
import math


def bilinear_interpolation(img, orig_w=320, orig_h=464):
    return cv2.resize(np.float32(img), dsize=(orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


def threshold(image):
    image = cv2.blur(image, (2, 2))
    ret, thresh = cv2.threshold(np.uint8(image), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = thresh.shape
    thresh = cv2.rectangle(thresh, (0, 0), (10, h), 0, -1)  # draw rect on left border
    thresh = cv2.rectangle(thresh, (0, 0), (w, 10), 0, -1)  # draw rect on top border
    thresh = cv2.rectangle(thresh, (w-10, 0), (w, h), 0, -1)  # draw rect on right border
    thresh = cv2.rectangle(thresh, (0, h - 10), (w, h), 0, -1)  # draw rect on bottom border
    return thresh


def connected_components(img):
    # gray = np.uint8(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    blur = 5
    img = np.uint8(cv2.blur(img, (blur, blur)))

    # getting mask with connectComponents
    output = cv2.connectedComponentsWithStats(img, )
    num_labels = output[0]
    labels = output[1]
    stats = output[2]

    blank_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    area_threshold = 500
    components = []
    for label in range(1, num_labels):
        mask = np.array(labels, dtype=np.uint8)
        mask[labels == label] = 255
        mask[labels != label] = 0

        if stats[label][cv2.CC_STAT_AREA] > area_threshold:
            components.append(np.array(mask))
            blank_image[labels == label] = [100, 100, 20 * label]

    return components


def gabor_kernel(theta):
    ksize = (10, 10)
    return cv2.getGaborKernel(ksize, sigma=2.5, theta=theta, lambd=5, gamma=0.5, ktype=cv2.CV_32F)


def compute_filter_avg(image, kernel, i, j):
    image_length = len(image)
    image_width = len(image[0])

    filtered = [[0] * kernel[1] for _ in range(kernel[0])]

    k_row = 0
    for row in range(i, i + kernel[0]):
        k_col = 0
        for col in range(j, j + kernel[1]):
            if 0 <= row < image_length and 0 <= col < image_width:
                filtered[k_row][k_col] = image[row][col]
            else:
                filtered[k_row][k_col] = float('-inf')
            k_col = k_col + 1
        k_row = k_row + 1
    return np.average(filtered)


def average_pooling(image, kernel, stride):
    length = len(image)
    width = len(image[0])
    new_image = []
    for i in range(0, length, stride[0]):
        row = []
        for j in range(0, width, stride[1]):
            row.append(compute_filter_avg(image, kernel, i, j))
            if j + kernel[1] >= width:
                break
        new_image.append(row)
        if i + kernel[0] >= length:
            break
    return np.array(new_image)


def apply_pyramid(img):
    base = img

    layer_1 = average_pooling(base, (2, 2), (2, 2))
    layer_1 = average_pooling(layer_1, (2, 2), (2, 2))
    layer_2 = average_pooling(layer_1, (2, 2), (2, 2))
    layer_3 = average_pooling(layer_2, (2, 2), (2, 2))

    pyramid = [layer_1, layer_2, layer_3]

    # Run Gabor Kernel
    # K should have arrays of sizes 4, 8, 8, 8
    K = [[], [], [], []]
    for h in range(0, 4):
        theta = h * math.pi / 4.
        g_kernel = gabor_kernel(theta)
        filtered_img = cv2.filter2D(base, cv2.CV_8UC3, g_kernel)
        K[0].append(filtered_img)

    for i in range(len(pyramid)):
        for h in range(0, 8):
            theta = h * math.pi / 8.
            g_kernel = gabor_kernel(theta)
            filtered_img = cv2.filter2D(base, cv2.CV_8UC3, g_kernel)
            gabor = bilinear_interpolation(filtered_img, base.shape[1], base.shape[0])
            K[i + 1].append(gabor)

    return K


def get_lines(gray_img):
    gray_img = cv2.blur(gray_img, (10, 10))
    out = apply_pyramid(gray_img)

    thresholded_imgs = [[threshold(im) for im in level] for level in out]
    ccs = []

    for level in thresholded_imgs:
        for im in level:
            ccs.extend(connected_components(im))

    return ccs