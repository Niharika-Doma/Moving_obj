import cv2
import numpy as np

# Initialize the video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide a video file path

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

while True:
    ret, frame = cap.read()
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

    if alert:
        cv2.putText(frame, "ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
