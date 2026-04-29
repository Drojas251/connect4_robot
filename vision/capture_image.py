import cv2

CAMERA_ID = 1
OUTPUT_PATH = "captured_board.jpg"


def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    print("Press SPACE to capture an image, or ESC/q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        cv2.imshow("Live Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            print("Exiting without saving.")
            break
        if key == 32:  # SPACE
            cv2.imwrite(OUTPUT_PATH, frame)
            print(f"Saved image to {OUTPUT_PATH}")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
