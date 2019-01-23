import numpy as np
import cv2
import os
import sys
import math

# Get relative path
dirname = os.path.dirname(__file__)
imageSetDir = os.path.join(dirname, 'image_set')
filename = os.path.join(imageSetDir, 'crayons.jpg')
print(filename)
img = cv2.imread(filename)
BLUE = 0
GREEN = 1
RED = 2

def getColourChannelMask(img, colour):
    return img[:,:,colour]

def createRedKernel(x, y):
    filter = np.zeros((x, y), dtype=np.int64)
    # indexing = start:end:step
    # 1::2 = start at index 1, go until end, steps of 2
    # ::2 = start at index 0, go until end, steps of 2
    filter[1::2, ::2] = 1 # even rows
    filter[::2, 1::2] = 1 # odd rows
    # Get average
    return filter/(x*y)

def createBlueKernel(x, y):
    filter = np.zeros((x, y), dtype=np.int64)
    filter[::2, ::2] = 1 # odd rows
    return filter/(x*y)

def createGreenKernel(x, y):
    filter = np.zeros((x, y), dtype=np.int64)
    filter[1::2, 1::2] = 1 # even rows
    return filter/(x*y)

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
# 1. Load mosaic image
# 2. Create the 3 colour channel masks according to pattern
# 3. Multiply the mosaic image with the masks -> extract output into variable
# 4. Design kernel to get nearest neighbour averages
# 5. Apply filter2D w/ kernel to outputs from step 3 -> extract each output to variable
# 6. Merge outputs from step 5 -> get demosaiced image
# 7. Calculate root square difference of *original* image with demosaiced image -> get artifacts
r_img = getColourChannelMask(img.copy(), RED)
g_img = getColourChannelMask(img.copy(), GREEN)
b_img = getColourChannelMask(img.copy(), BLUE)

r_kernel = createRedKernel(3,3)
g_kernel = createGreenKernel(3,3)
b_kernel = createBlueKernel(3,3)

r_dst = cv2.filter2D(r_img, -1, r_kernel)
g_dst = cv2.filter2D(g_img, -1, g_kernel)
b_dst = cv2.filter2D(b_img, -1, b_kernel)

b_sqdiff = imgRootSquaredDifference(img, b_dst, BLUE)
g_sqdiff = imgRootSquaredDifference(img, g_dst, GREEN)
r_sqdiff = imgRootSquaredDifference(img, r_dst, RED)

dst = b_sqdiff + g_sqdiff + r_sqdiff

cv2.imshow('img', img)
cv2.imshow('dst', dst)

# Difference between original and demosaiced images
diff = img - dst
cv2.imshow('diff', diff)

cv2.waitKey(0)
cv2.destroyAllWindows()