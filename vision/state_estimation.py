import cv2
import numpy as np


CAMERA_ID = 1

# Tune these if needed
LOWER_BLUE = np.array([80, 40, 40])
UPPER_BLUE = np.array([140, 255, 255])
WARP_WIDTH = 700
WARP_HEIGHT = 600


def order_points(pts):
    """
    Order points as:
    top-left, top-right, bottom-right, bottom-left
    """
    pts = np.array(pts, dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


def find_blue_board(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, mask

    largest = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(largest)
    if area < 1000:
        return None, mask

    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    box = order_points(box)

    return box, mask


def warp_board(frame, corners):
    dst = np.array([
        [0, 0],
        [WARP_WIDTH - 1, 0],
        [WARP_WIDTH - 1, WARP_HEIGHT - 1],
        [0, WARP_HEIGHT - 1],
    ], dtype="float32")

    H = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(frame, H, (WARP_WIDTH, WARP_HEIGHT))
    return warped


def draw_grid(warped):
    cell_w = WARP_WIDTH // 7
    cell_h = WARP_HEIGHT // 6

    out = warped.copy()

    for col in range(7):
        for row in range(6):
            cx = col * cell_w + cell_w // 2
            cy = row * cell_h + cell_h // 2

            cv2.circle(out, (cx, cy), 8, (0, 255, 0), -1)
            cv2.rectangle(
                out,
                (col * cell_w, row * cell_h),
                ((col + 1) * cell_w, (row + 1) * cell_h),
                (255, 255, 255),
                1
            )

    return out


def nothing(x):
    pass


def main():
    cap = cv2.VideoCapture(CAMERA_ID)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    # --- HSV slider window ---
    cv2.namedWindow("HSV Sliders")
    cv2.createTrackbar("H low", "HSV Sliders", 80, 179, nothing)
    cv2.createTrackbar("S low", "HSV Sliders", 40, 255, nothing)
    cv2.createTrackbar("V low", "HSV Sliders", 40, 255, nothing)
    cv2.createTrackbar("H high", "HSV Sliders", 140, 179, nothing)
    cv2.createTrackbar("S high", "HSV Sliders", 255, 255, nothing)
    cv2.createTrackbar("V high", "HSV Sliders", 255, 255, nothing)
    # -------------------------

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        # --- Read HSV slider values ---
        lh = cv2.getTrackbarPos("H low", "HSV Sliders")
        ls = cv2.getTrackbarPos("S low", "HSV Sliders")
        lv = cv2.getTrackbarPos("V low", "HSV Sliders")
        hh = cv2.getTrackbarPos("H high", "HSV Sliders")
        hs = cv2.getTrackbarPos("S high", "HSV Sliders")
        hv = cv2.getTrackbarPos("V high", "HSV Sliders")
        lower = np.array([lh, ls, lv])
        upper = np.array([hh, hs, hv])
        # -----------------------------

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # --- Filter out blue pixels and show result ---
        frame_no_blue = frame.copy()
        frame_no_blue[mask > 0] = (0, 0, 0)
        cv2.imshow("No Blue Pixels", frame_no_blue)
        # ---------------------------------------------

        # --- Find blue board corners using current mask ---
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        corners = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area >= 1000:
                rect = cv2.minAreaRect(largest)
                box = cv2.boxPoints(rect)
                corners = order_points(box)
        # -------------------------------------------------

        if corners is not None:
            corners_int = corners.astype(int)

            cv2.polylines(
                display,
                [corners_int],
                isClosed=True,
                color=(0, 255, 0),
                thickness=3
            )

            for i, p in enumerate(corners_int):
                x, y = p
                cv2.circle(display, (x, y), 8, (0, 0, 255), -1)
                cv2.putText(
                    display,
                    str(i),
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            warped = warp_board(frame, corners)
            warped_grid = draw_grid(warped)

            cv2.imshow("Warped Board", warped_grid)

        cv2.imshow("Camera", display)
        cv2.imshow("Blue Mask", mask)

        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()