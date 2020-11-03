import math

import cv2
import numpy as np
from PIL import Image


def cut(image, block_size, padding, width, height):
    row_num = math.ceil(height / block_size)
    col_num = math.ceil(width / block_size)  # 行数和列数
    image_list = []
    for j in range(0, row_num):
        b = j * block_size
        d = (j + 1) * block_size + 2 * padding
        for i in range(0, col_num):
            a = i * block_size
            c = (i + 1) * block_size + 2 * padding
            image_block = image[b:d, a:c]
            image_list.append(image_block)
    return image_list


def unpadding(image, padding):
    width, height = image.shape[1], image.shape[0]
    image = image[padding:height - padding, padding:width - padding]
    return image


def padding(image, padding):
    image = np.array(image)
    new_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REFLECT)
    return Image.fromarray(new_image)


def restore(images, block_size, width, height):
    row_num = math.ceil(height / block_size)
    col_num = math.ceil(width / block_size)
    images = [images[i:i + col_num] for i in range(0, len(images), col_num)]
    row_images = []
    for i, row in enumerate(images):
        row_image = None
        for j, item in enumerate(row):
            if j == 0:
                row_image = item
            else:
                row_image = cv2.hconcat([row_image, item])
        row_images.append(row_image)
    image = None
    for i, row in enumerate(row_images):
        if i == 0:
            image = row
        else:
            image = cv2.vconcat([image, row])
    return image
