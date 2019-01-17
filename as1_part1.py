import numpy as np
import cv2

def createRedFilter(x, y):
    filter = np.zeros((x, y),dtype=int)
    # indexing = start:end:step
    # 1::2 = start at index 1, go until end, steps of 2
    # ::2 = start at index 0, go until end, steps of 2
    filter[1::2,::2] = 1 # even rows
    filter[::2,1::2] = 1 # odd rows
    return filter

def createBlueFilter(x, y):
    filter = np.zeros((x, y),dtype=int)
    filter[::2,::2] = 1 # odd rows
    return filter

def createGreenFilter(x, y):
    filter = np.zeros((x, y),dtype=int)
    filter[1::2,1::2] = 1 # even rows
    return filter

print("Red filter:")
print(createRedFilter(5,5))
print("\nBlue filter:")
print(createBlueFilter(5,5))
print("\nGreen filter:")
print(createGreenFilter(5,5))