import cv2
import time
import numpy as np

# Load face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

prev_cx, prev_cy = None, None
prev_time = time.time()

direction = "Center"
speed = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Assume one main face
    for (x, y, w, h) in faces:
        # Draw green box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Compute center
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # If previous center exists, compute movement
        if prev_cx is not None:
            dx = cx - prev_cx
            dy = cy - prev_cy

            # Compute direction
            if abs(dx) > abs(dy):
                if dx > 10:
                    direction = "Right"
                elif dx < -10:
                    direction = "Left"
                else:
                    direction = "Center"
            else:
                if dy > 10:
                    direction = "Down"
                elif dy < -10:
                    direction = "Up"
                else:
                    direction = "Center"

            # Compute speed (pixels/second)
            curr_time = time.time()
            dt = curr_time - prev_time
            dist = np.sqrt(dx**2 + dy**2)
            speed = dist / dt if dt > 0 else 0
            prev_time = curr_time

        # Update previous
        prev_cx, prev_cy = cx, cy

        break  # Only first face

    # Display direction and speed
    cv2.putText(frame, f"Direction: {direction}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Speed: {speed:.2f} px/s", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Face Direction Tracker", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
