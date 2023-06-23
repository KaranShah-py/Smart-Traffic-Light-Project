import cv2
import numpy as np
from time import sleep

min_width = 80
min_height = 80

offset = 6

line_pos = 550

delay = 60

detections = []
car_count = 0

def get_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

cap = cv2.VideoCapture(r"C:\Users\karan\Python\Projects\Traffic light Final Year Project\video.mp4")
subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame1 = cap.read()
    time = float(1/delay)
    sleep(time)
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 5)
    img_sub = subtractor.apply(blur)
    dilate = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilate = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    dilate = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame1, (25, line_pos), (700, line_pos), (0, 127, 255), 3)
    for (i, contour) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        valid_contour = (w >= min_width) and (h >= min_height)
        if not valid_contour:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center = get_center(x, y, w, h)
        detections.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        for (x, y) in detections:
            if y < (line_pos + offset) and y > (line_pos - offset):
                car_count += 1
                cv2.line(frame1, (25, line_pos), (700, line_pos), (0, 127, 255), 3)
                detections.remove((x, y))
                print("Car detected: " + str(car_count))

    cv2.putText(frame1, "VEHICLE COUNT: " + str(car_count), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Original Video", frame1)
    cv2.imshow("Detection", dilate)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
