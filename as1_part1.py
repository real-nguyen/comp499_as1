import numpy as np
import cv2
import os
import sys
import math

# Get relative path
dirname = os.path.dirname(__file__)
imageSetDir = os.path.join(dirname, 'image_set')
imageName = 'pencils'
img_mosaic = cv2.imread(os.path.join(imageSetDir, imageName + '_mosaic.bmp'), cv2.IMREAD_GRAYSCALE)
img_original = cv2.imread(os.path.join(imageSetDir, imageName + '.jpg'))
img_rows = img_mosaic.shape[0]
img_cols = img_mosaic.shape[1]
BLUE = 0
GREEN = 1
RED = 2

def createColourMask(rows, cols, colour):
    mask = np.zeros((rows, cols), dtype=np.float32)
    # indexing = start:end:step
    # 1::2 = start at index 1, go until end, steps of 2
    # ::2 = start at index 0, go until end, steps of 2
    if colour == BLUE:
        mask[::2, ::2] = 1 # odd rows
    elif colour == GREEN:
        mask[1::2, 1::2] = 1 # even rows
    elif colour == RED:
        mask[1::2, ::2] = 1 # even rows
        mask[::2, 1::2] = 1 # odd rows
    return mask

def mergeChannels(b_chan, g_chan, r_chan):
    img = np.zeros((img_rows, img_cols, 3), dtype=np.uint8)
    for row in range(img_rows):
        for col in range(img_cols):
            img[row, col, BLUE] = b_chan[row, col]
            img[row, col, GREEN] = g_chan[row, col]
            img[row, col, RED] = r_chan[row, col]
    return img

def imgRootSquaredDifference(img1, img2):
    # img1 should be original image
    # img2 should be colour masked and filtered image
    newImg = img1.copy()
    # Replace all elements with 0
    newImg[:] = 0
    
    for row in range(img_rows):
        for col in range(img_cols):
            for channel in range(3):
                pixel1 = int(img1[row, col, channel])
                pixel2 = int(img2[row, col, channel])
                # Need to convert to int because pixels are stored as bytes by default
                # Squaring the values would overflow and calculate the wrong value
                newImg[row, col, channel] = math.sqrt((pixel1 - pixel2) ** 2)
    return newImg

b_mask = createColourMask(img_rows, img_cols, BLUE)
g_mask = createColourMask(img_rows, img_cols, GREEN)
r_mask = createColourMask(img_rows, img_cols, RED)

b_mult = img_mosaic * b_mask
g_mult = img_mosaic * g_mask
r_mult = img_mosaic * r_mask

# Get average of 2 or 4 nearest neighbours
# Blue and green channels share the same kernel
bg_kernel = np.array([[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]])
r_kernel = np.array([[0, 0.25, 0], [0.25, 1, 0.25], [0, 0.25, 0]])

# Use numpy's clip method in case some values go over 255
b_avg = np.clip(cv2.filter2D(b_mult, -1, bg_kernel), 0, 255)
g_avg = np.clip(cv2.filter2D(g_mult, -1, bg_kernel), 0, 255)
r_avg = np.clip(cv2.filter2D(r_mult, -1, r_kernel), 0, 255)

img_demosaic = mergeChannels(b_avg, g_avg, r_avg)

cv2.imshow('img_mosaic', img_mosaic)
cv2.imshow('img_original', img_original)
cv2.imshow('img_demosaic', img_demosaic)

diff = imgRootSquaredDifference(img_original, img_demosaic)
cv2.imshow('diff', diff)
cv2.waitKey(0)
cv2.destroyAllWindows()