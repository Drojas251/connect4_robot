import cv2
import numpy as np
from pupil_apriltags import Detector


CAMERA_INDEX = 1
MAX_FRAMES = 200
TAG_FAMILY = "tag36h11"

ROI_TAGS = [1, 2, 3, 9]

TAG_PAIRS = [
    (5, 4),
    (10, 13),
    (14, 12),
]


def get_tag_corner_points(detection):
    corners = np.array(detection.corners, dtype=np.float32)
    y_sorted = corners[np.argsort(corners[:, 1])]

    top_two = y_sorted[:2]
    bottom_two = y_sorted[2:]

    top_left, top_right = top_two[np.argsort(top_two[:, 0])]
    bottom_left, bottom_right = bottom_two[np.argsort(bottom_two[:, 0])]

    return {
        "top_left": top_left,
        "top_right": top_right,
        "bottom_right": bottom_right,
        "bottom_left": bottom_left,
    }


def clamp_box(x_min, y_min, x_max, y_max, frame_shape):
    h, w = frame_shape[:2]

    x_min = max(0, min(w - 1, int(x_min)))
    x_max = max(0, min(w, int(x_max)))
    y_min = max(0, min(h - 1, int(y_min)))
    y_max = max(0, min(h, int(y_max)))

    return x_min, y_min, x_max, y_max


def compute_middle_rows_from_tags(detections):
    detections_by_id = {d.tag_id: d for d in detections}
    middle_rows = []

    for left_id, right_id in TAG_PAIRS:
        if left_id not in detections_by_id or right_id not in detections_by_id:
            print(f"Missing pair: ({left_id}, {right_id})")
            continue

        left = np.array(detections_by_id[left_id].center, dtype=np.float32)
        right = np.array(detections_by_id[right_id].center, dtype=np.float32)

        center = (left + right) / 2.0
        tag_vec = right - left

        left_slot = left - 0.5 * tag_vec
        right_slot = right + 0.5 * tag_vec

        slot_spacing_vec = center - left_slot

        far_left_slot = left_slot - slot_spacing_vec
        far_right_slot = right_slot + slot_spacing_vec

        row_slots = [
            far_left_slot,
            left_slot,
            center,
            right_slot,
            far_right_slot,
        ]

        middle_rows.append(row_slots)

    return middle_rows


def compute_full_5x5_grid(middle_rows):
    if len(middle_rows) != 3:
        raise RuntimeError(
            f"Expected 3 middle rows from tags, got {len(middle_rows)}"
        )

    row0 = np.array(middle_rows[0], dtype=np.float32)
    row1 = np.array(middle_rows[1], dtype=np.float32)
    row2 = np.array(middle_rows[2], dtype=np.float32)

    top_spacing = row1 - row0
    bottom_spacing = row2 - row1

    top_row = row0 - top_spacing
    bottom_row = row2 + bottom_spacing

    full_grid = [
        top_row,
        row0,
        row1,
        row2,
        bottom_row,
    ]

    return full_grid


def draw_full_grid(crop, full_grid):
    for row_idx, row in enumerate(full_grid):
        for col_idx, pt in enumerate(row):
            pt_int = tuple(np.round(pt).astype(int))

            cv2.circle(crop, pt_int, 10, (255, 255, 0), 2)

            cv2.putText(
                crop,
                f"R{row_idx}C{col_idx}",
                (pt_int[0] + 8, pt_int[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
            )

    # Draw horizontal row lines
    for row in full_grid:
        pts = np.round(row).astype(int)
        for i in range(len(pts) - 1):
            cv2.line(
                crop,
                tuple(pts[i]),
                tuple(pts[i + 1]),
                (0, 255, 0),
                1,
            )

    # Draw vertical column lines
    for col_idx in range(5):
        for row_idx in range(4):
            p1 = tuple(np.round(full_grid[row_idx][col_idx]).astype(int))
            p2 = tuple(np.round(full_grid[row_idx + 1][col_idx]).astype(int))
            cv2.line(crop, p1, p2, (0, 255, 255), 1)

    return crop


def main():
    detector = Detector(
        families=TAG_FAMILY,
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}.")

    found_frame = None
    found_detections_by_id = None

    print(f"Searching up to {MAX_FRAMES} frames for ROI tags {ROI_TAGS}...")

    for frame_idx in range(MAX_FRAMES):
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(gray)
        detections_by_id = {d.tag_id: d for d in detections}

        visible_ids = sorted(detections_by_id.keys())
        print(f"Frame {frame_idx}: detected tags {visible_ids}")

        if all(tag_id in detections_by_id for tag_id in ROI_TAGS):
            print(f"Found all ROI tags in frame {frame_idx}.")
            found_frame = frame.copy()
            found_detections_by_id = detections_by_id
            break

    cap.release()

    if found_frame is None:
        print(f"Did not find all ROI tags {ROI_TAGS} within {MAX_FRAMES} frames.")
        return

    tag1_corners = get_tag_corner_points(found_detections_by_id[1])
    tag2_corners = get_tag_corner_points(found_detections_by_id[2])
    tag3_corners = get_tag_corner_points(found_detections_by_id[3])
    tag9_corners = get_tag_corner_points(found_detections_by_id[9])

    roi_points = np.array(
        [
            tag1_corners["top_left"],
            tag2_corners["top_right"],
            tag3_corners["bottom_right"],
            tag9_corners["bottom_left"],
        ],
        dtype=np.float32,
    )

    x_min = np.min(roi_points[:, 0])
    y_min = np.min(roi_points[:, 1])
    x_max = np.max(roi_points[:, 0])
    y_max = np.max(roi_points[:, 1])

    x_min, y_min, x_max, y_max = clamp_box(
        x_min, y_min, x_max, y_max, found_frame.shape
    )

    crop = found_frame[y_min:y_max, x_min:x_max].copy()

    if crop.size == 0:
        print("Crop is empty. Check ROI tag locations.")
        return

    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop_detections = detector.detect(crop_gray)

    print("Tags detected inside crop:", sorted([d.tag_id for d in crop_detections]))

    middle_rows = compute_middle_rows_from_tags(crop_detections)

    if len(middle_rows) != 3:
        print("Could not compute all 3 tagged rows.")
        return

    full_grid = compute_full_5x5_grid(middle_rows)

    crop_with_grid = draw_full_grid(crop, full_grid)

    print("\nFull 5x5 grid slot centers:")
    for r, row in enumerate(full_grid):
        for c, pt in enumerate(row):
            print(f"R{r}C{c}: ({pt[0]:.1f}, {pt[1]:.1f})")

    debug_frame = found_frame.copy()

    cv2.polylines(
        debug_frame,
        [roi_points.astype(np.int32)],
        isClosed=True,
        color=(0, 255, 0),
        thickness=2,
    )

    cv2.imshow("Full Frame ROI", debug_frame)
    cv2.imshow("ROI Crop With 5x5 Slot Grid", crop_with_grid)

    print("\nPress any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()