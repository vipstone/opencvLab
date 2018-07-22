#
# coding=utf-8
# description: file operation functions
#

import os
import cv2


IMAGE_SUFFIX = ["jpg", "jpeg", "png"]


def is_image(image_name):
    return any([image_name.lower().endswith(suffix) for suffix in IMAGE_SUFFIX])


def get_images_data(train_path, image_flag=cv2.IMREAD_COLOR):
    """
    loading training images or test images.
    :param train_path:
    :param image_flag:
    :return:
    """
    if not os.path.isdir(train_path):
        raise NotADirectoryError(train_path + " is not a dir path.")

    data = []

    for path in os.listdir(train_path):
        name = path
        path = os.path.join(train_path, path)
        images = []
        data.append({
            "label": name,
            "data": images
        })
        for image_name in os.listdir(path):
            if not is_image(image_name):
                continue
            images.append(cv2.imread(os.path.join(path, image_name), image_flag))
    return data


if __name__ == '__main__':
    pass
