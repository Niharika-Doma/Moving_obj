import cv2
import numpy as np
import pygame
from twilio.rest import Client
import time

# Twilio configuration
account_sid = 'ACb10de3c28e38fa2b7edbd500b94dc89c'
auth_token = 'edd272172fe0c482aaf1704053252d7c'
twilio_phone_number = '+17626758416'  # Your Twilio phone number
recipient_phone_number = '+919959068500'  # The phone number to receive alerts

client = Client(account_sid, auth_token)

# Initialize pygame mixer
pygame.mixer.init()
pygame.mixer.music.load("C:/Users/msvar/Desktop/Junk/Group_Project/beep-04.wav")

# Initialize the video capture
cam = cv2.VideoCapture(0)  # Use 0 for webcam, or provide a video file path

# Create a background subtractor
backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Define the region of interest (ROI) - example coordinates
roi = [(100, 100), (500, 100), (500, 400), (100, 400)]

def point_inside_polygon(x, y, poly):
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def send_sms_alert():
    try:
        message = client.messages.create(
            body="Alert! Object detected in the monitored area for 15 consecutive frames.",
            from_=twilio_phone_number,
            to=recipient_phone_number
        )
        print(f"SMS alert sent. SID: {message.sid}")
    except Exception as e:
        print(f"Failed to send SMS: {str(e)}")

# Flag to track if music is currently playing
music_playing = False

# Variables for object detection and SMS alert
consecutive_detections = 0
alert_sent = False

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Apply background subtraction
    fgMask = backSub.apply(frame)

    # Threshold the mask
    thresh = cv2.threshold(fgMask, 244, 255, cv2.THRESH_BINARY)[1]

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw ROI
    cv2.polylines(frame, [np.array(roi)], True, (0, 255, 0), 2)
   
    alert = False
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Adjust this value to filter out small movements
            x, y, w, h = cv2.boundingRect(contour)
            if point_inside_polygon(x + w//2, y + h//2, roi):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                alert = True
                break  # Exit the loop as soon as we find one valid contour
                 
    if alert:
        consecutive_detections += 1
        cv2.putText(frame, f"ALERT! Count: {consecutive_detections}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Play music if it's not already playing
        if not music_playing:
            pygame.mixer.music.play(-1)  # -1 means loop indefinitely
            music_playing = True

        # Send SMS alert if 15 consecutive detections and alert not sent yet
        if consecutive_detections >= 15 and not alert_sent:
            send_sms_alert()
            alert_sent = True

    else:
        consecutive_detections = 0
        alert_sent = False
        # Stop music if it's playing and there's no alert
        if music_playing:
            pygame.mixer.music.stop()
            music_playing = False

    # Display the frame
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()  # Ensure music stops when the program ends
pygame.mixer.quit()
