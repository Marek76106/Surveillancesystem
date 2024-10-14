import sys
import cv2
import numpy as np

# Load full body classifier
fullbody_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
# Initialize KCF tracker
tracker = cv2.TrackerKCF_create()
# Initialize bounding box of person
bbox = None
def main():
    # Open video file
    cap = cv2.VideoCapture('2.mp4')
    #cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Find first frame in video
    ret, frame = cap.read()
    
    # Initialize the frame counter for tracking
    tracking_counter = 0
    while True:
        # read the frame
        ret, frame = cap.read()
        if not ret:
            break
        #resize the frame
        frame = cv2.resize(frame,None,fx=0.4,fy=0.4,interpolation=cv2.INTER_NEAREST)
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # If the object is being tracked
        if tracking_counter > 0:
            # If there is a tracker, update tracker
            if bbox is not None:
                success, bbox = tracker.update(frame)
                if success:
                    (x, y, w, h) = [int(i) for i in bbox]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    tracking_counter += 1
                else:
                    tracking_counter = 0
                    bbox = None
            # if 20 frames have been tracked
            if tracking_counter >= 25:
                tracking_counter = 0
        else:
        # detect objects in the frame
            fullbodies = fullbody_cascade.detectMultiScale(gray, 1.10, 2, minSize=(90, 150))
            if len(fullbodies) > 0:
                (x, y, w, h) = fullbodies[0]
                bbox = (x, y, w, h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, bbox)
                tracking_counter += 1
            # show the frame
        cv2.imshow("Video", frame)
            # if the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    # cleanup the camera and close any open windows
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()