#!/usr/bin/env python
import sys

# Import OpenCV and Numpy
import cv2
import numpy as np



def trackObjects(frame,cx,cy,cw,ch):
    # Specify region of interest (ROI) of the car
    #global track
    if(cx==0):
        print("zero")
    else:
        x,y,w,h = cx,cy,cw,ch
        #199, 400, 908, 150
        trackWindow = (x, y, w, h)
        roi = frame[x:x+h, y:y+w]
        # Convert the roi to HSV color space
        roiHSV = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Perform color filtering of roi to obtain mask 
        lower = np.array([0, 0, 254])
        upper = np.array([0, 0, 255])
        mask = cv2.inRange(roiHSV, lower, upper)
        
        # Calculate histogram
        roiHist = cv2.calcHist([roiHSV], [0], mask, [180], [0,180])
        # Normalize the histogram
        cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
        # Define termination criteria
        term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, 10)
        
        # Convert frame to HSV color space
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Perform back projection on frame and roi
        dst = cv2.calcBackProject([frameHSV], [0], roiHist, [0, 180], 1)
        # Perform Meanshift or Camshift algorithm
            
            
        #ret, trackWindow = cv2.meanShift(dst, trackWindow, term)
        ret, trackWindow = cv2.CamShift(dst, trackWindow, term)
        points = cv2.boxPoints(ret)
        points = np.int0(points)
        image = cv2.polylines(frame, [points], True, 255, 2)
            
        # Retrieve and draw new tracking window
        x, y, w, h = trackWindow
        cv2.rectangle(frame, (x,y), (x+w,y+h), 255, 2)
            
            
        # Display the image with tracked object.
        cv2.imshow('Tracked Object', frame)
        #track=="netrack"
        #print("raditrack")
    
def findObjects(image, path):
    
    # Convert image to grayscale
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Initialize classifier's object
    classifier = cv2.CascadeClassifier(path)

    # Perform detection on the image, experiment with parameters for better results
    objects = classifier.detectMultiScale(image_grayscale, 1.10, 2, minSize=(150, 150))

    # Code below is already added, no change is needed, but it is a good idea to study the code
    # Write if no objects were found
    cx,cy,cw,ch = 0, 0, 0, 0
    if len(objects) == 0:
        print ("No objects found")
    # Draw rectangles for each object
    else:
        for (x,y,w,h) in objects:
        
            cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
            cx=int(x)
            cy=int(y)
            cw=int(w)
            ch=int(h)
        #detections.append([cx,cy,cw,ch])
        
        #print(track)
        #track="track"
        #print(detections)
    # Show the image
    #detections.append(int[x,y,w,h])
    
    #cv2.imshow("Found Objects", image)
    return cx,cy,cw,ch
    
    
    
def main():
    # Open video device
    video = cv2.VideoCapture('2.mp4')
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)
    # Loop 50 times
    global track
    track="netrack"
    detections = []
    for i in range(length):
        # Read the frame
        ret, frame = video.read()
        #Extract Region of Intrest
        #roi = frame[132: 1042, 263: 1095]
        # Perform operation every 10th frame
        if i % 17 == 0:
        #while track==False:
            # Write the frame
            path = 'haarcascade_fullbody.xml'
            x, y, w, h = findObjects(frame, path)
            #cv2.imshow("FRAME", frame)
            #print(x,y,w,h)
            
        #while track=="track":
        trackObjects(frame,x,y,w,h)
        cv2.waitKey(30)
    # Release the video
    video.release()
    

if __name__ == '__main__':
    main()