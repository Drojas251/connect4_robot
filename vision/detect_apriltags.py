import cv2
from pupil_apriltags import Detector

# Initialize detector
detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
)

# Open camera
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = detector.detect(gray)

    for d in detections:
        # Draw corners
        for i in range(4):
            pt1 = tuple(map(int, d.corners[i]))
            pt2 = tuple(map(int, d.corners[(i+1)%4]))
            cv2.line(frame, pt1, pt2, (0,255,0), 2)

        # Center
        cx, cy = map(int, d.center)
        cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)

        # Tag ID
        cv2.putText(frame, f"ID {d.tag_id}",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255,0,0), 2)

    cv2.imshow("AprilTag Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()