import cv2
import imutils
import numpy as np
import math		   
from PIL import Image

def erode(mask, kernel_size=(3,3), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    eroded_mask = mask.copy()
    for _ in range(iterations):
        eroded_mask = np.minimum.reduce([eroded_mask[i:mask.shape[0]-kernel_size[0]+i+1, j:mask.shape[1]-kernel_size[1]+j+1] for i in range(kernel_size[0]) for j in range(kernel_size[1])])
    return eroded_mask

# Function for dilation
def dilate(mask, iterations=2):
    dilated_mask = mask.copy()
    for _ in range(iterations):
        dilated_mask[1:-1, 1:-1] |= dilated_mask[:-2, 1:-1]
        dilated_mask[1:-1, 1:-1] |= dilated_mask[2:, 1:-1]
        dilated_mask[1:-1, 1:-1] |= dilated_mask[1:-1, :-2]
        dilated_mask[1:-1, 1:-1] |= dilated_mask[1:-1, 2:]
    return dilated_mask

def equalize_hist(image):
    # Calculate histogram
    hist, bins = np.histogram(image.flatten(), 256, [0,256])
    # Calculate cumulative distribution function (CDF)
    cdf = hist.cumsum()
    # Normalize CDF
    cdf_normalized = cdf / float(cdf.max())
    # Perform histogram equalization
    equalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalized * 255).reshape(image.shape)
    return equalized_image.astype(np.uint8)

def hisEqulColor(img):
    ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    channels=cv2.split(ycrcb)
    cv2.equalizeHist(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    return cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR)

cap = cv2.VideoCapture(0)

while 1:
    ret, frame = cap.read()    
    orig_img = cv2.GaussianBlur(frame, (11, 11), 3)
    #eq_img = hisEqulColor(frame)
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)

    greenLower = (29, 86, 18)
    greenUpper = (93, 255, 255)

    # masking out the green color
    mask = ((np.logical_and(np.all(img >= greenLower, axis = -1), np.all(img <= greenUpper, axis = -1))).astype(np.uint8))*255
    mask = erode(mask)
    mask = dilate(mask)

    cog_x = 0
    cog_y = 0
    total = 0

    indices = np.argwhere(mask > 120)

    # Sum up x and y indices
    cog_x = np.sum(indices[:, 1])
    cog_y = np.sum(indices[:, 0])

    # Count total number of elements
    total = indices.shape[0]

    if (total):
        cog_x /= total
        cog_y /= total
        cv2.circle(frame, (int(cog_x),int(cog_y)), 2, (255, 0, 0), 1)

    #cv2.imshow('live feed', eq_img)
    cv2.imshow('orig', frame)
    cv2.imshow('mask', mask)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()