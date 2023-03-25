import cv2
import math

# Load the cascade files for detecting the knee
knee_cascade: object = cv2.CascadeClassifier('haarcascade_mcs_upperbody.xml')

# Load the video
cap = cv2.VideoCapture('knee_video.mp4')

while True:
    # Read a frame from the video
    ret, img = cap.read()

    # Convert the frame to grayscale
    gray: None = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the knee in the grayscale image
    knees = knee_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through each detected knee
    for (x, y, w, h) in knees:
        # Draw a rectangle around the knee
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Calculate the center of the knee
        center_x = x + w / 2
        center_y = y + h / 2

        # Draw a circle at the center of the knee
        cv2.circle(img, (int(center_x), int(center_y)), 3, (0, 0, 255), -1)

        # Calculate the angle of the knee
        angle = math.atan2(center_y - y, center_x - x) * 180.0 / math.pi

        # Display the angle on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(round(angle, 2)), (int(center_x), int(center_y)), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the image
    cv2.imshow('Knee Pose Estimation', img)

    # Wait for a key press
    k = cv2.waitKey(30) & 0xff

    # If the user presses 'q', exit the loop
    if k == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()