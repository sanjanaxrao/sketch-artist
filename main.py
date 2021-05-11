import cv2
from face_localization import find_face, create_ratio
from extract_lines import get_lines
from post_processing import thinning
from generate_output import draw_lines
from shading import shade


def crop_image(gray_img, face_localization=True, ratio=1.45):
    if face_localization:
        x, y, w, h = find_face(gray_img, ratio)
        return gray_img[y:(y+h), x:(x+w)]
    else:
        img_w, img_h = gray_img.shape
        x, y, w, h = create_ratio(img_w, img_h, 0, 0, img_w, img_h, ratio)
        return gray_img[y:(y+h), x:(x+w)]


def load_img(filepath, face_localization):
    # read and crop image
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = crop_image(gray, face_localization)
    return img


if __name__ == "__main__":
    filepaths = ["test_images/lady.png", 'test_images/other_lady.png', 'test_images/ner.png']
    file = filepaths[0]
    gray_img = load_img(file, face_localization=False)
    lines = get_lines(gray_img)  # 1d arr black images with white lines
    print('lines extracted')
    processed_lines = thinning(lines)
    print('lines thinned')

    shading = shade(gray_img)  # 2d arr of pts per spline
    print('shading completed')
    print('drawing output ' + str(file))
    print()
    draw_lines(processed_lines, shading, file, save_img=False)

