#
# coding=utf-8
# description:
#
import cv2


def show_image(img, width=None, height=None, win_name="Image", wait_time=0):
    cv2.imshow(win_name, resize(img, width, height))
    cv2.waitKey(wait_time) & 0XFF


def resize(img, width, height, interpolation=cv2.INTER_AREA):
    size = None

    if width is None and height is None:
        size = (img.shape[1], img.shape[0])
    elif width is not None:
        ratio = width / float(img.shape[1])
        height = int(img.shape[0] * ratio)
        size = (width, height)
    elif height is not None:
        ratio = height / float(img.shape[0])
        width = int(img.shape[1] * ratio)
        size = (width, height)

    return cv2.resize(img, size, interpolation=interpolation)


def draw_rect(img, rect, color=(0, 255, 0), line_width=1):
    cv2.rectangle(img, rect[:2], rect[2:], color, line_width)


def put_text(img, text, rect, color=(0, 255, 0), thickness=0.8):
    cv2.putText(img, text, rect[:2], cv2.FONT_HERSHEY_COMPLEX_SMALL, thickness, color)


if __name__ == '__main__':
    pass
