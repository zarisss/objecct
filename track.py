import cv2
import numpy as np

# Initialize the video capture
capture = cv2.VideoCapture("traf.mp4")

# Create a MultiTracker object
multi_tracker = cv2.legacy.MultiTracker_create()

# Read the first frame
ret, frame = capture.read()

# List of all bounding boxes and paths
bounding_boxes = []
paths = []
prev_centers = []
fps = capture.get(cv2.CAP_PROP_FPS)

def add_new_objects(frame):
    global bounding_boxes, paths, prev_centers
    new_boxes = cv2.selectROIs("Add New Objects", frame, fromCenter=False, showCrosshair=True)
    if len(new_boxes) > 0:
        for bbox in new_boxes:
            multi_tracker.add(cv2.legacy.TrackerCSRT_create(), frame, tuple(bbox))
            bounding_boxes.append(bbox)
            paths.append([])
            prev_centers.append(None)

# Initial object selection
add_new_objects(frame)

while True:
    ret, frame = capture.read()
    if not ret:
        break

    # Update the trackers with the new frame
    ret, boxes = multi_tracker.update(frame)

    for i, bbox in enumerate(boxes):
        x, y, w, h = [int(v) for v in bbox]
        center = (x + w // 2, y + h // 2)

        # Speed and direction
        if prev_centers[i] is not None:
            dx = center[0] - prev_centers[i][0]
            dy = center[1] - prev_centers[i][1]
            distance = np.sqrt(dx**2 + dy**2)
            speed = distance / fps
            cv2.putText(frame, f"Speed: {speed:.2f} px/s", (10, 30 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            direction = ""
            if abs(dx) > abs(dy):
                direction = "Right" if dx > 0 else "Left"
            else:
                direction = "Down" if dy > 0 else "Up"
            cv2.putText(frame, f"Direction: {direction}", (10, 45 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Draw bounding box and path
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        paths[i].append(center)
        for j in range(1, len(paths[i])):
            if paths[i][j - 1] is None or paths[i][j] is None:
                continue
            cv2.line(frame, paths[i][j - 1], paths[i][j], (255, 0, 0), 2)

        prev_centers[i] = center

        # Alert if object goes out of frame
        frame_height, frame_width, _ = frame.shape
        if x < 0 or y < 0 or x + w > frame_width or y + h > frame_height:
            cv2.putText(frame, f"Object {i+1} out of frame!", (10, 70 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Object Tracking", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('a'):
        # Add new objects while running
        add_new_objects(frame)

capture.release()
cv2.destroyAllWindows()
