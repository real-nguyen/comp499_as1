import numpy as np
import cv2
import os
import sys
import math

# Get relative path
dirname = os.path.dirname(__file__)
imageSetDir = os.path.join(dirname, 'image_set')
filename = os.path.join(imageSetDir, 'crayons_mosaic.bmp')
img = cv2.imread(filename)
img_Y = img.shape[0]
img_X = img.shape[1]
BLUE = 0
GREEN = 1
RED = 2

def createColourMask(x, y, colour):
    mask = np.zeros((x, y), dtype=np.float32)
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

def imgRootSquaredDifference(img1, img2, colour):
    # img1 should be original image
    # img2 should be colour masked and filtered image
    newImg = img1.copy()
    # Replace all elements with 0
    newImg[:] = 0
    
    for row in range(0, img1.shape[0]):
        for col in range(0, img1.shape[1]):
            try:
                # Need to convert to int because pixels are stored as bytes by default
                # Squaring the values would overflow and calculate the wrong value
                val = math.sqrt((int(img1[row, col, colour]) - int(img2[row, col])) ** 2)
                newImg[row, col, colour] = val
            except RuntimeWarning:
                print('error')
    return newImg

# TODO: REDO EVERYTHING
# ~~1. Load mosaic image~~ DONE
# ~~2. Create the 3 colour channel masks according to pattern~~ DONE
# 3. Multiply the mosaic image with the masks -> extract output into variable
# 4. Design kernel to get nearest neighbour averages
# 5. Apply filter2D w/ kernel to outputs from step 3 -> extract each output to variable
# 6. Merge outputs from step 5 -> get demosaiced image
# 7. Calculate root square difference of *original* image with demosaiced image -> get artifacts
b_mask = createColourMask(img_X, img_Y, BLUE)
g_mask = createColourMask(img_X, img_Y, GREEN)
r_mask = createColourMask(img_X, img_Y, RED)


# cv2.imshow('img', img)
# cv2.imshow('dst', dst)

# # Difference between original and demosaiced images
# diff = img - dst
# cv2.imshow('diff', diff)

# cv2.waitKey(0)
# cv2.destroyAllWindows()