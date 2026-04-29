import cv2
import numpy as np

INPUT_PATH = "captured_board.jpg"

# Default HSV blue range (adjust as needed)
LOWER_BLUE = np.array([80, 40, 40])
UPPER_BLUE = np.array([140, 255, 255])


def nothing(x):
    pass

def main():
    img = cv2.imread(INPUT_PATH)
    if img is None:
        print(f"Could not read image: {INPUT_PATH}")
        return

    cv2.namedWindow("HSV Sliders")
    cv2.createTrackbar("H low", "HSV Sliders", LOWER_BLUE[0], 179, nothing)
    cv2.createTrackbar("S low", "HSV Sliders", LOWER_BLUE[1], 255, nothing)
    cv2.createTrackbar("V low", "HSV Sliders", LOWER_BLUE[2], 255, nothing)
    cv2.createTrackbar("H high", "HSV Sliders", UPPER_BLUE[0], 179, nothing)
    cv2.createTrackbar("S high", "HSV Sliders", UPPER_BLUE[1], 255, nothing)
    cv2.createTrackbar("V high", "HSV Sliders", UPPER_BLUE[2], 255, nothing)

    while True:
        lh = cv2.getTrackbarPos("H low", "HSV Sliders")
        ls = cv2.getTrackbarPos("S low", "HSV Sliders")
        lv = cv2.getTrackbarPos("V low", "HSV Sliders")
        hh = cv2.getTrackbarPos("H high", "HSV Sliders")
        hs = cv2.getTrackbarPos("S high", "HSV Sliders")
        hv = cv2.getTrackbarPos("V high", "HSV Sliders")
        lower = np.array([lh, ls, lv])
        upper = np.array([hh, hs, hv])

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        img_no_blue = img.copy()
        img_no_blue[mask > 0] = (0, 0, 0)

        # --- Edge filter (Canny) on mask only ---
        edges = cv2.Canny(mask, 100, 200)
        cv2.imshow("Edges (Mask)", edges)
        # --------------------------------------

        cv2.imshow("Original", img)
        cv2.imshow("Blue Mask", mask)
        cv2.imshow("No Blue Pixels", img_no_blue)
        cv2.imshow("HSV Sliders", np.zeros((1, 400), dtype=np.uint8))  # Just to keep window open

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
