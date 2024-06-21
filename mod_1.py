import cv2
import time
import pygame

#loading music for alert sound
pygame.mixer.init()
pygame.mixer.music.load("C:/Users/Dell/Desktop/Group project/beep-04.wav")



# Create a background subtractor object
backSub = cv2.createBackgroundSubtractorMOG2()

# Capture video from webcam or a video file
cap = cv2.VideoCapture(0)  # 0 for webcam, path to video file otherwise
time.sleep(1)

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

alert = False

while True:
    ret, frame = cap.read()
    text = "Object is Stable"

    if not ret:
        break

    # Apply background subtraction
    fg_mask = backSub.apply(frame)

    # Threshold the foreground mask
    thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)[1]

    # Find contours of moving objects
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    '''if len(contours)>0:
        #plays the alert
        winsound.PlaySound(alert,winsound.SND_FILENAME)'''

    if any(cv2.contourArea(cnt)>500 for cnt in contours):
        if not alert:
            pygame.mixer.music.play()
            alert = True
        else:
            alert = False

    # Draw contours and bounding boxes around moving objects
    count= 0
    for cnt in contours:
        # Filter out small contours (noise)
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Moving object is detected"
            print(text)
            count = count+1
            print(count)
    cv2.putText(frame, text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

    # Display the frame with detected objects
    cv2.imshow("Frame", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release capture and close windows
cap.release()
cv2.destroyAllWindows()
