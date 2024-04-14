import cv2
import numpy as np
from kalmanfilter import KalmanFilter

def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (0, 255, 0), 3, 1)
    cv2.putText(img, "Tracking", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


capture = cv2.VideoCapture("MESSY.mp4")
tracker = cv2.legacy.TrackerCSRT_create()
ret, frame = capture.read()
bounding_box = cv2.selectROI("Tracking", frame, False)
tracker.init(frame, bounding_box)

while True:
    timer = cv2.getTickCount()
    ret, frame = capture.read()
    if not ret:
        break

    edge = cv2.Canny(frame, 100, 100)
    ret, bounding_box = tracker.update(edge)
    print(bounding_box)
    if ret:
        drawBox(frame, bounding_box)
    else:
        cv2.putText(frame, "lost", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(frame, str(int(fps)), (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Tracking", frame)
    cv2.imshow("TRANS", edge)

    if cv2.waitKey(30) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
