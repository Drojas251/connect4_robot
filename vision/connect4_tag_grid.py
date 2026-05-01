import cv2
import numpy as np
from dataclasses import dataclass
from pupil_apriltags import Detector


@dataclass
class HoleCenter:
    row: int
    col: int
    roi_xy: tuple[int, int]
    frame_xy: tuple[int, int]


@dataclass
class Connect4GridResult:
    bbox_xyxy: tuple[int, int, int, int]
    roi_image: np.ndarray
    holes: list[HoleCenter]


class Connect4TagGridDetector:
    def __init__(
        self,
        tag_family="tag36h11",
        roi_tags=(1, 2, 3, 9),
        tag_pairs=((5, 4), (10, 13), (14, 12)),
        nthreads=4,
        quad_decimate=1.0,
    ):
        self.roi_tags = list(roi_tags)
        self.tag_pairs = list(tag_pairs)

        self.detector = Detector(
            families=tag_family,
            nthreads=nthreads,
            quad_decimate=quad_decimate,
            quad_sigma=0.0,
            refine_edges=1,
        )

    def detect_from_frame(self, frame: np.ndarray) -> Connect4GridResult | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)
        detections_by_id = {d.tag_id: d for d in detections}

        if not all(tag_id in detections_by_id for tag_id in self.roi_tags):
            return None

        bbox, roi_points = self._compute_roi_bbox(frame, detections_by_id)
        x_min, y_min, x_max, y_max = bbox

        roi = frame[y_min:y_max, x_min:x_max].copy()
        if roi.size == 0:
            return None

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_detections = self.detector.detect(roi_gray)

        middle_rows = self._compute_middle_rows_from_tags(roi_detections)

        if len(middle_rows) != 3:
            return None

        full_grid = self._compute_full_5x5_grid(middle_rows)

        holes = []
        for row_idx, row in enumerate(full_grid):
            for col_idx, pt in enumerate(row):
                roi_x, roi_y = np.round(pt).astype(int)
                frame_x = roi_x + x_min
                frame_y = roi_y + y_min

                holes.append(
                    HoleCenter(
                        row=row_idx,
                        col=col_idx,
                        roi_xy=(int(roi_x), int(roi_y)),
                        frame_xy=(int(frame_x), int(frame_y)),
                    )
                )

        return Connect4GridResult(
            bbox_xyxy=bbox,
            roi_image=roi,
            holes=holes,
        )

    def _compute_roi_bbox(self, frame, detections_by_id):
        tag1 = self._get_tag_corner_points(detections_by_id[1])
        tag2 = self._get_tag_corner_points(detections_by_id[2])
        tag3 = self._get_tag_corner_points(detections_by_id[3])
        tag9 = self._get_tag_corner_points(detections_by_id[9])

        roi_points = np.array(
            [
                tag1["top_left"],
                tag2["top_right"],
                tag3["bottom_right"],
                tag9["bottom_left"],
            ],
            dtype=np.float32,
        )

        x_min = np.min(roi_points[:, 0])
        y_min = np.min(roi_points[:, 1])
        x_max = np.max(roi_points[:, 0])
        y_max = np.max(roi_points[:, 1])

        bbox = self._clamp_box(x_min, y_min, x_max, y_max, frame.shape)
        return bbox, roi_points

    def _compute_middle_rows_from_tags(self, detections):
        detections_by_id = {d.tag_id: d for d in detections}
        middle_rows = []

        for left_id, right_id in self.tag_pairs:
            if left_id not in detections_by_id or right_id not in detections_by_id:
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

    def _compute_full_5x5_grid(self, middle_rows):
        row0 = np.array(middle_rows[0], dtype=np.float32)
        row1 = np.array(middle_rows[1], dtype=np.float32)
        row2 = np.array(middle_rows[2], dtype=np.float32)

        top_spacing = row1 - row0
        bottom_spacing = row2 - row1

        top_row = row0 - top_spacing
        bottom_row = row2 + bottom_spacing

        return [
            top_row,
            row0,
            row1,
            row2,
            bottom_row,
        ]

    @staticmethod
    def _get_tag_corner_points(detection):
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

    @staticmethod
    def _clamp_box(x_min, y_min, x_max, y_max, frame_shape):
        h, w = frame_shape[:2]

        x_min = max(0, min(w - 1, int(x_min)))
        x_max = max(0, min(w, int(x_max)))
        y_min = max(0, min(h - 1, int(y_min)))
        y_max = max(0, min(h, int(y_max)))

        return x_min, y_min, x_max, y_max