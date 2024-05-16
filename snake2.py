import cv2
import imutils
import numpy as np
import math
from tkinter import messagebox as mb
import random

# Snake game in Python


WIN_SCORE = 10
MAX_OBSTACLES = 5
MIN_DISTANCE_BW_OBSTACLES = 60
MIN_DISTANCE_FOOD_OBSTACLE = 60
LENGTH_INCREASE_FROM_FOOD = 20
STARTING_LENGTH = 70
SNAKE_COLOR = (0, 0, 255)
SNAKE_THICKNESS = 12
GREEN_LOWER_THRESHOLD = (29, 86, 18)
GREEN_UPPER_THRESHOLD = (93, 255, 255)
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

# Distance function
def distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

# Check if a point on overlaps an image on the frame
def checkPointOverlapImage(pointLocation, imageLocation, imageSize):
    return (imageLocation[0] - imageSize[0]//2 < pointLocation[0] < imageLocation[0] + imageSize[0]//2 and \
        imageLocation[1] - imageSize[1]//2 < pointLocation[1] < imageLocation[1] + imageSize[1]//2)

# Draw image over frame
def drawImageOverFrame(frame, image, imageLocation):
    imageSize = image.shape[1], image.shape[0]
    frame[(imageLocation[1] - (imageSize[1]//2)) : (imageLocation[1] + (imageSize[1]//2)), 
            (imageLocation[0] - (imageSize[0]//2)) : (imageLocation[0] + (imageSize[0]//2))] = image

# Class for running the game logic
class SnakeGame:
    def __init__(self, foodImagePath, obstacleImagePath) -> None:
        self.points = []
        self.lengths = []
        self.totalLength = STARTING_LENGTH
        self.currentLength = 0
        self.obstacleLocations = []
        self.obstacleCount = MAX_OBSTACLES
        self.previousHead = 0, 0
        self.score = 0
        self.gameOver = False

        self.obstacleImage = cv2.imread(obstacleImagePath, cv2.IMREAD_UNCHANGED)
        self.obstacleImageSize = self.obstacleImage.shape[1], self.obstacleImage.shape[0]
        self.generateObstacles()

        self.foodImage = cv2.imread(foodImagePath, cv2.IMREAD_UNCHANGED)
        self.foodImageSize = self.foodImage.shape[1], self.foodImage.shape[0]
        self.foodLocation = 0, 0
        self.generateRandomFoodLocation()
    
    # Randomly generate obstacles
    def generateObstacles(self):
        self.obstacleLocations = []
        while len(self.obstacleLocations) < self.obstacleCount:
            temp_point = (random.randint(50, VIDEO_WIDTH-50), random.randint(50, VIDEO_HEIGHT-50))
            if all(distance(temp_point,p) >= MIN_DISTANCE_BW_OBSTACLES for p in self.obstacleLocations):
                self.obstacleLocations.append(temp_point)

    # Randomly generate food location
    def generateRandomFoodLocation(self):
        while True:
            self.foodLocation = random.randint(50, VIDEO_WIDTH-50), random.randint(50, VIDEO_HEIGHT-50)
            if all(distance(self.foodLocation, obstacle) > MIN_DISTANCE_FOOD_OBSTACLE for obstacle in self.obstacleLocations):
                break
    
    # Draw snake, food and obstacles
    def drawObjects(self, image):
        # Draw snake
        if self.points:
            for i, _ in enumerate(self.points):
                if i != 0:
                    cv2.line(image, self.points[i], self.points[i-1], SNAKE_COLOR, SNAKE_THICKNESS)
            cv2.circle(image, self.points[-1], int(SNAKE_THICKNESS/2), SNAKE_COLOR, cv2.FILLED)
        
        # Draw food
        drawImageOverFrame(image, self.foodImage, self.foodLocation)

        # Draw obstacles
        for location in self.obstacleLocations:
            drawImageOverFrame(image, self.obstacleImage, location)
        
        return image

    # Reset game variables
    def reset(self):
        self.points = []
        self.lengths = []
        self.totalLength = STARTING_LENGTH
        self.currentLength = 0
        self.obstacleLocations = []
        self.obstacleCount = MAX_OBSTACLES
        self.previousHead = 0, 0
        self.score = 0
        self.gameOver = False
        self.generateObstacles()
        self.generateRandomFoodLocation()

    # Called when game is finished
    def gameFinished(self, frame, win: bool):
        if win:
            cv2.putText(frame, 'YOU WIN!!', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
        else:
            cv2.putText(frame, 'GAME OVER!!', (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
        cv2.imshow("Camera", frame)
        res = mb.askquestion('Exit Application', 'Play again?')       
        if res == 'yes' :
            self.reset()
        else:
            global playGame
            playGame = False
        
    # Actual game loop is implemented here
    def updateSnake(self, image, NewHeadLocation):
        self.points.append(NewHeadLocation)
        dist_to_last_head = distance(self.previousHead, NewHeadLocation)
        self.lengths.append(dist_to_last_head)
        self.currentLength += dist_to_last_head
        self.previousHead = NewHeadLocation

        # Reduce current length if more than total length
        if self.currentLength > self.totalLength:
            for len_i, length in enumerate(self.lengths):
                self.currentLength -= length
                self.points.pop(len_i)
                self.lengths.pop(len_i)
                if self.currentLength < self.totalLength:
                    break
        
        # Check if food was eaten
        if checkPointOverlapImage(NewHeadLocation, self.foodLocation, self.foodImageSize):
            self.totalLength += LENGTH_INCREASE_FROM_FOOD
            self.score += 1
            self.generateRandomFoodLocation()
        
        # Display snake, food, and obstacles
        image = self.drawObjects(image)

        # Display score
        cv2.putText(image, 'Score :' + str(self.score), (450, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 203), 2)
        
        # Check collision with obstacles
        for location in self.obstacleLocations:
            if checkPointOverlapImage(NewHeadLocation, location, self.obstacleImageSize):
                self.gameFinished(image, False)
        
        # Check collision with snake body
        check_pts = np.array(self.points[:-6], np.int32)
        check_pts = check_pts.reshape((-1,1,2))
        cv2.polylines(image, [check_pts], False, (255,0,0), 0)
        minimum_dist = cv2.pointPolygonTest(check_pts, NewHeadLocation, True)
        if -1 <= minimum_dist<= 1:
            self.gameFinished(image, False)

        # Check win condition
        if self.score >= WIN_SCORE:
            self.gameFinished(image, True)
        
        return image

# Function for erosion
def erode(mask, kernel_size=(3,3), iterations=1):
    eroded_mask = mask.copy()
    for _ in range(iterations):
        eroded_mask = np.minimum.reduce([
            eroded_mask[i:mask.shape[0]-kernel_size[0]+i+1, j:mask.shape[1]-kernel_size[1]+j+1] 
            for i in range(kernel_size[0]) 
            for j in range(kernel_size[1])])
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

# 2D point for welzl's algorithm
class Point_W:
    def __init__(self, X=0, Y=0) -> None:
        self.X=X
        self.Y=Y

# Circle for welzl'a algorithm
class Circle_W:
    def __init__(self, c=Point_W(), r=0) -> None:        
        self.C=c
        self.R=r

# Distance between 2 point_w
def dist_2points(a: Point_W, b: Point_W) -> float:
    return math.sqrt((a.X - b.X)**2 + (a.Y - b.Y)**2)

# Check if point is inside circle
def point_inside_circle(c: Circle_W, p: Point_W) -> bool:
    return dist_2points(c.C, p) <= c.R

# Check if points P lie in circle
def is_circle_valid(c: Circle_W, P: list[Point_W]):
    for p in P:
        if (not point_inside_circle(c, p)):
            return False
    return True

# Get circle for trivial cases: len(P) <= 3
def get_circle_trivial(P: list[Point_W]):
    if not P:
        return Circle_W() 
    elif (len(P) == 1):
        return Circle_W(P[0], 0) 
    elif (len(P) == 2):
        return get_circle_2points(P[0], P[1])
    for i in range(3):
        for j in range(i + 1,3):
            c = get_circle_2points(P[i], P[j])
            if (is_circle_valid(c, P)):
                return c
    return get_circle_3points(P[0], P[1], P[2])

# Returns smallest circle with the 2 points on boundary
def get_circle_2points(A: Point_W, B: Point_W):
    Center = Point_W((A.X + B.X) / 2.0, (A.Y + B.Y) / 2.0 )
    return Circle_W(Center, dist_2points(Center, A))

# Returns smallest circle with the 3 points on boundary
def get_circle_3points(A: Point_W, B: Point_W, C: Point_W):
    baX = B.X - A.X
    baY = B.Y - A.Y
    caX = C.X - A.X
    caY = C.Y - A.Y
    tB = baX**2 + baY**2
    tC = caX**2 + caY**2
    tD = baX*caY - baY*caX
    Center = Point_W(A.X + (caY*tB - baY*tC)/(tD*2), A.Y + (baX*tC - caX*tB)/(tD*2))
    return Circle_W(Center, dist_2points(Center, A))

# Recursive welzl helper function
def welzl_helper(P: list[Point_W], R: list[Point_W], n: int):
    if (n == 0 or len(R) == 3) :
        return get_circle_trivial(R)
    r_index = random.randint(0,n-1)
    random_point = P[r_index]
    P[r_index],P[n-1] = P[n-1],P[r_index]
    temp_C = welzl_helper(P, R.copy(), n-1)
    if (point_inside_circle(temp_C, random_point)):
        return temp_C
    R.append(random_point)
    return welzl_helper(P, R.copy(), n-1)

# Applies Welzl's algorithm on points P
def welzl(P: list[Point_W]):
    P_copy = P.copy()
    random.shuffle(P_copy)
    return welzl_helper(P_copy, [], len(P_copy))

playGame = True
game = SnakeGame("apple.jpg", "hurdle1small.jpg")
cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 720)


while playGame:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Image preprocessing
    img = cv2.GaussianBlur(frame, (11, 11), 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((np.logical_and(np.all(img >= GREEN_LOWER_THRESHOLD, axis = -1), np.all(img <= GREEN_UPPER_THRESHOLD, axis = -1))).astype(np.uint8))*255
    mask = erode(mask)
    mask = dilate(mask)

    # Find contours in image
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if len(contours) > 0:
        ball_contour = max(contours, key=cv2.contourArea)
        # Assuming ball_contour is the largest contour obtained from the binary mask image

        # Convert the contour to a list of points
        points = [Point_W(x[0][0], x[0][1]) for x in ball_contour]

        # Find the minimum enclosing circle using Welzl's algorithm
        min_enclosing_circle = welzl(points)
        center = (int(min_enclosing_circle.C.X), int(min_enclosing_circle.C.Y))
        radius = int(min_enclosing_circle.R)

        # If radius big enough, run game loop
        if radius > 10:
            frame = game.updateSnake(frame, center)
        # otherwise, only display game objects
        else:
            frame = game.drawObjects(frame)
    else:
        frame = game.drawObjects(frame)
    
    cv2.imshow("Camera", frame)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
