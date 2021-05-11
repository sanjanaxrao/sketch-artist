import cv2

def create_ratio(img_h, img_w, face_x, face_y, face_w, face_h, ratio):
    # change y and h to be desired ratio
    final_x = face_x
    final_y = face_y
    final_width = face_w
    final_height = int(face_w * ratio)

    if final_height > img_h:
        final_width = int(img_h / ratio) # make width smaller so ratio holds
        final_height = img_h # set height to image height
        final_x = face_x + int((face_w - final_width) / 2) # recenter top-left x for new width
    else:
        # recenter top-left y for new height
        final_y = face_y + int((face_h - final_height) / 2)

    return final_x, final_y, final_width, final_height

def find_face(gray_img, ratio):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)

    # Draw the rectangle around each face
    dims = (0, 0, 0, 0)
    for face in faces:
        (x, y, w, h) = face

        # save the largest rectangle found
        if w*h > dims[2]*dims[3]:
            img_h, img_w = gray_img.shape
            dims = create_ratio(img_h, img_w, x, y, w, h, ratio)

    # draw rectangle on image
    # x, y, w, h = dims
    # cv2.rectangle(gray_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # cv2.imshow('img w/ rectangles', gray_img)
    # cv2.waitKey(0)

    return dims


if __name__ == "__main__":
    filename = "test_images/test.jpg"
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    find_face(gray, 1.45)