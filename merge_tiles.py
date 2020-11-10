import sys
import os
import numpy as np
from PIL import Image
import cv2

path = "E:\\camelyon16\\TrainingData\\merge_tile\\patches\\Mask/"
dirs = os.listdir(path)

images = [Image.open(x) for x in ['Test1.jpg', 'Test2.jpg', 'Test3.jpg']]
widths, heights = zip(*(i.size for i in images))
num_columns =1
num_rows=2

total_width = sum(widths)
max_height = max(heights)

new_im = Image.new('GRAY', (total_width, max_height))

x_offset = 0
i = 0
j = 0
image_size = 0
for i in range(num_columns):
    for j in range(num_rows):

        if os.path.isfile(path + image_size):
            im = Image.open(path + image_size)
        else:
            im = image = Image.new('RGB', (256, 256))
        image_size = image_size + 1

        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

scale_percent = 80  # percent of original size
width = int(new_im.shape[1] * scale_percent / 100)
height = int(new_im.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(new_im, dim, interpolation=cv2.INTER_AREA)

print('Resized Dimensions : ', resized.shape)

cv2.imshow("Resized image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
resized.save('test.jpg')