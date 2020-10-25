from PIL import Image
import numpy as np
import cv2
import math

def restore(images, block_size, width, height):
    row_num = math.ceil(height / block_size)
    col_num = math.ceil(width / block_size)
    images = [images[i:i+col_num] for i in range(0, len(images), col_num)]
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


images = []
for i in range(1, 10):
    image = Image.open("in%d.jpg"%i)
    image = np.array(image)
    images.append(image)
image = restore(images, block_size=1000, width=3000, height=3000)
image = Image.fromarray(image)
image.save("final_mosaic.jpg")