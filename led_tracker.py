import cv2
import numpy as np

CAM_INDEX = 0  # If not using the limelight camera.
USING_LIMELIGHT = False
LIMELIGHT_URL = "http://10.0.127.18:5802"
MIN_AREA = 5


# change the window name depending on which camera is used.
window_name = "Limelight Stream" if USING_LIMELIGHT else "Camera Stream"
stream_type = LIMELIGHT_URL if USING_LIMELIGHT else CAM_INDEX

# start like this for green
lower_color = np.array([35, 100, 100])
upper_color = np.array([85, 255, 255])


capture = cv2.VideoCapture(stream_type)

if not capture.isOpened():
    print(f"Error: Cannot open stream at {stream_type}")
    exit()

while True:
    ret, frame = capture.read()
    if not ret:
        print("Failed to get frame")
        break

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_color, upper_color)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    led_centers = []

    for c in contours:
        area = cv2.contourArea(c)

        if area > MIN_AREA:
            x, y, w, h = cv2.boundingRect(c)

            # Draw a rectangle around the led (on the original frame)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Find the center point of each detected led and put it in an array
            center_x = x + (w//2)
            center_y = y + (h//2)
            led_centers.append((center_x, center_y))

            # Draw a circle for the center point in each detected led
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

    cv2.imshow(window_name, frame)
    cv2.imshow("Mask", mask)

    # Wait until either the escape key is pressed or 'q'.
    key = cv2.waitKey(1) & 0XFF
    if key == ord('q') or key == 27:
        break


capture.release()
cv2.destroyAllWindows()
