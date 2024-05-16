import cv2
import imutils
import numpy as np
import math		   
from PIL import Image
import tkinter as tk 
from tkinter import * 
from tkinter import messagebox as mb 

# Snake game in Python

score = 0
max_score = 9
list_capacity = 0
max_lc = 20
l = []
flag = 0
apple_x = None
apple_y = None
center = None
hurdles = []



#Defining functions to read hurdle image
'''yet to decide'''
def read_image(path):
    original_image = cv2.imread('hurdle1.jpg', 1)
    resized_image = cv2.resize(original_image, (30, 30))
    blackandwht = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    alpha, _, _ = otsu_threshold(blackandwht)
    b, g, r = cv2.split(resized_image)
    rgba = [b, g, r, alpha]
    final_img = cv2.merge(rgba, 4)
    final_img = cv2.cvtColor(final_img, cv2.COLOR_RGBA2RGB)
    return final_img



def otsu_threshold(im_in):
    image_data = im_in.flatten()
    bins = np.arange(0,100,1)

    hist = {bin: 0 for bin in bins}
    
    for i in image_data:
        for bin in bins:
            if i <= bin:
                hist[bin] +=1
                break

    hist_arr = np.array(list(hist.values()))
    bins_arr = np.array(list(hist.keys()))

    pdf = hist_arr / sum(hist_arr)

    cdf = [sum(pdf[:i+1]) for i in range(len(pdf))]

    cum_int = np.cumsum(np.arange(100) * pdf)
    mu = {}

    total_mean = cum_int[-1]
    max_var, best_thresh = 0, 0
    for thresh in range(1,100):
        fore = cdf[thresh]
        back = 1 - fore

        if fore == 0 or back == 0:
            continue
        mean_fore = cum_int[thresh] / fore
        mean_back = (total_mean - cum_int[thresh]) / back

        mu_fb = fore * back * (mean_back - mean_fore)**2
        mu[thresh] = mu_fb
        if mu_fb > max_var:
            max_var = mu_fb
            best_thresh = thresh

    thresh_image = np.zeros_like(im_in)
    thresh_image[im_in > best_thresh] = 99

    return thresh_image, best_thresh, mu




# Define hurdle positions
# Function to generate random apple and hurdle positions
def generate_hurdles(frame):
    num_hurdles = 5
    global hurdles
    # min_dist = 100
    if not hurdles:
        # Generate random hurdle positions
        hurdles = [(np.random.randint(30, frame.shape[1] - 30), np.random.randint(30, frame.shape[0] - 30)) for _ in range(num_hurdles)]

# distance function
def dist(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)



def calculate_moments(contour):
    # Calculate moments manually
    M = {}
    M['m00'] = len(contour)
    m10 = 0
    m01 = 0
    for point in contour:
        x, y = point[0]
        m10 += x
        m01 += y
    M['m10'] = m10
    M['m01'] = m01
    # Check for divide by zero error
    if M['m00'] == 0:
        M['m00'] = 1
    # Calculate centroid
    centroid_x = int(M['m10'] / M['m00'])
    centroid_y = int(M['m01'] / M['m00'])
    return M, (centroid_x, centroid_y)


								   
											  

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

																						
def convolution(f, I):
    filter_ht, filter_wdt = f.shape
    image_ht, image_wdt, image_depth = I.shape

    pad_ht = filter_ht // 2
    pad_wdt = filter_wdt // 2
    pad_img = np.pad(I, ((pad_ht, pad_ht), (pad_wdt, pad_wdt), (0, 0)), mode='constant')

    im_conv = np.zeros_like(I, dtype=float)

    for i in range(image_ht):
        for j in range(image_wdt):
            for k in range(image_depth):
                roi = pad_img[i:i+filter_ht, j:j+filter_wdt, k]
                im_conv[i, j, k] = np.sum(roi * f)
    return im_conv

def gaussian_filter(sigma, filter_size):
    # Ensure filter size is odd
    filter_size = filter_size + 1 if filter_size % 2 == 0 else filter_size

    # Calculate range of values
    x = np.arange(-filter_size // 2, filter_size // 2 + 1)

    # Calculate Gaussian kernel
    kernel = np.exp(-0.5 * (x / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
    kernel /= np.sum(kernel)

    return kernel											 
		
# Load hurdle image
hurdle_image = read_image('hurdle_image.png')  # Provide the path to your hurdle image


cap = cv2.VideoCapture(0)
				  				
res = 'no'

ret, frame = cap.read()
generate_hurdles(frame)

while 1:
    ret, frame = cap.read()
		  					   
    img = imutils.resize(frame.copy(), width=600)
    img = cv2.GaussianBlur(img, (11, 11), 0)
	
	#cannot add manual gaussian blur as it takes more time to process each frame and filter it. Though, the trials for it are mentioned below.
    '''# img = gaussian_blur(img, 11, 1)
    sigma = 1
    filter_size = 11
    gaussian_kernel = gaussian_filter(sigma, filter_size)
    gaussian_kernel = gaussian_kernel.reshape(-1,1)
    # if isinstance(gaussian_kernel, np.ndarray):
    #     print("yes")
    # else:
    #     print("no")
    # # print("gaussian_kernel", gaussian_kernel.dtype)
    img = convolution(gaussian_kernel, img)
    #print("image shape", img.shape)
    img = img.astype(np.uint8)
    # img_x = gaussian_blur(img, 11, 1)'''																																									  
										  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Generate random apple and hurdle positions if needed

    if apple_x is None or apple_y is None:
        # assigning random coefficients for apple coordinates
        min_dist = 50
        distance = 10
        while distance <= min_dist:
            apple_x = np.random.randint(30, frame.shape[0] - 30)
            apple_y = np.random.randint(100, 350)
            if all(np.sqrt((apple_x - hurdle[0])**2 + (apple_y - hurdle[1])**2) > min_dist for hurdle in hurdles):
                break
    cv2.circle(frame, (apple_x, apple_y), 6, (0, 0, 255), -1)
    

    # Display hurdles
    for hurdle_pos in hurdles:
        # Draw hurdle image at hurdle position
        hurdle_x, hurdle_y = hurdle_pos
        frame[hurdle_y:hurdle_y + hurdle_image.shape[0], hurdle_x:hurdle_x + hurdle_image.shape[1]] = hurdle_image

    # change this range according to your need
    greenLower = (29, 86, 18)
    greenUpper = (93, 255, 255)

    # masking out the green color
    mask = ((np.logical_and(np.all(img >= greenLower, axis = -1), np.all(img <= greenUpper, axis = -1))).astype(np.uint8))*255
    mask = erode(mask)
    #mask = cv2.erode(mask, None, iterations=2)
    # Dilate mask using custom function
    mask = dilate(mask)
    #mask = cv2.dilate(mask, None, iterations=2)

    # find contours
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) > 0:
        ball_cont = max(cnts, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(ball_cont)

        M, center = calculate_moments(ball_cont)
        
        # M = cv2.moments(ball_cont)
        # center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        if radius > 10:
            cv2.circle(frame, center, 4, (0, 0, 255), -1)

            if len(l) > list_capacity:
                l = l[1:]

            if prev_c and (dist(prev_c, center) > 3.5):
                l.append(center)

            apple = (apple_x, apple_y)
            if dist(apple, center) < 5:
                score += 1
                if score == max_score:
                    flag = 1
                list_capacity += 1
                apple_x = None
                apple_y = None

    for i in range(1, len(l)):
        if l[i - 1] is None or l[i] is None:
            continue
        r, g, b = np.random.randint(0, 255, 3)

        cv2.line(frame, l[i], l[i - 1], (int(r), int(g), int(b)), thickness=int(len(l) / max_lc + 2) + 6)

    cv2.putText(frame, 'Score :' + str(score), (450, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 203), 2)
    if flag == 1:
        cv2.putText(frame, 'YOU WIN !!', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 3)
        cv2.imshow('live feed', frame)
        res = mb.askquestion('Exit Application', 'Retry?')       
        if res == 'yes' :
            score = 0
            list_capacity = 0
            max_lc = 20
            l = []
            flag = 0
            center = None
            res = 'no'
            continue
              
        else : 
            cv2.waitKey(1000) 
            break
            #mb.showinfo('Return', 'Returning to main application') 
            # Delay before closing the window
        

    # Check collision with hurdles
    for hurdle_pos in hurdles:
        if center is not None and dist(center, hurdle_pos) < 15:
            flag = -1
            break

    # Game over condition
    if flag == -1:
        cv2.putText(frame, 'GAME OVER !!', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
        cv2.imshow('live feed', frame)
        res = mb.askquestion('Exit Application', 'Retry?')       
        if res == 'yes' :
            score = 0
            list_capacity = 0
            max_lc = 20
            l = []
            flag = 0
            center = None
            res = 'no'
            continue   
        else : 
            cv2.waitKey(1000) 
            break
    cv2.imshow('live feed', frame)
    cv2.imshow('mask', mask)

    prev_c = center

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
