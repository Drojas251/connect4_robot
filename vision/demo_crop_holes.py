import cv2
import numpy as np

from connect4_tag_grid import Connect4TagGridDetector


CAMERA_INDEX = 1
MAX_FRAMES = 200

HOLE_CROP_RADIUS = 18


def circular_crop_from_roi(roi, center_xy, radius):
    cx, cy = center_xy

    h, w = roi.shape[:2]

    x_min = max(0, cx - radius)
    x_max = min(w, cx + radius)
    y_min = max(0, cy - radius)
    y_max = min(h, cy + radius)

    patch = roi[y_min:y_max, x_min:x_max].copy()

    if patch.size == 0:
        return None

    patch_h, patch_w = patch.shape[:2]

    local_cx = cx - x_min
    local_cy = cy - y_min

    mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
    cv2.circle(mask, (local_cx, local_cy), radius, 255, -1)

    circular_patch = cv2.bitwise_and(patch, patch, mask=mask)

    return circular_patch


def make_hole_display_grid(hole_images, cell_size=80):
    canvas = np.zeros((5 * cell_size, 5 * cell_size, 3), dtype=np.uint8)

    for row, col, img in hole_images:
        if img is None:
            continue

        resized = cv2.resize(img, (cell_size, cell_size))

        y0 = row * cell_size
        y1 = y0 + cell_size
        x0 = col * cell_size
        x1 = x0 + cell_size

        canvas[y0:y1, x0:x1] = resized

        cv2.putText(
            canvas,
            f"R{row}C{col}",
            (x0 + 5, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

    return canvas


def draw_debug_grid(roi, holes):
    debug = roi.copy()

    for hole in holes:
        cv2.circle(debug, hole.roi_xy, HOLE_CROP_RADIUS, (255, 255, 0), 2)
        cv2.circle(debug, hole.roi_xy, 3, (0, 0, 255), -1)

        cv2.putText(
            debug,
            f"{hole.row},{hole.col}",
            (hole.roi_xy[0] + 6, hole.roi_xy[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
        )

    return debug


def main():
    grid_detector = Connect4TagGridDetector()

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    result = None
    source_frame = None

    print(f"Searching up to {MAX_FRAMES} frames...")

    for frame_idx in range(MAX_FRAMES):
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame.")
            continue

        result = grid_detector.detect_from_frame(frame)

        visible_msg = "FOUND" if result is not None else "not found"
        print(f"Frame {frame_idx}: grid {visible_msg}")

        if result is not None:
            source_frame = frame.copy()
            break

    cap.release()

    if result is None:
        print("Could not detect full Connect 4 grid.")
        return

    print("\nROI bbox:")
    print(result.bbox_xyxy)

    print("\nHole centers:")
    for hole in result.holes:
        print(
            f"R{hole.row} C{hole.col} | "
            f"roi={hole.roi_xy} | frame={hole.frame_xy}"
        )

    roi = result.roi_image

    hole_images = []

    for hole in result.holes:
        cropped_circle = circular_crop_from_roi(
            roi,
            hole.roi_xy,
            HOLE_CROP_RADIUS,
        )

        hole_images.append((hole.row, hole.col, cropped_circle))

    hole_grid_display = make_hole_display_grid(hole_images)
    roi_debug = draw_debug_grid(roi, result.holes)

    x_min, y_min, x_max, y_max = result.bbox_xyxy

    frame_debug = source_frame.copy()
    cv2.rectangle(
        frame_debug,
        (x_min, y_min),
        (x_max, y_max),
        (0, 255, 0),
        2,
    )

    cv2.imshow("Original Frame With ROI BBox", frame_debug)
    cv2.imshow("ROI With Hole Circles", roi_debug)
    cv2.imshow("Only Cropped Circular Hole Pixels", hole_grid_display)

    print("\nPress any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()