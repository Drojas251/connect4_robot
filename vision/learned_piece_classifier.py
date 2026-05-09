from pathlib import Path
import joblib
import numpy as np

from connect4_robot.game_engine.board import Cell
from .train_piece_classifier import extract_features


_DEFAULT_MODEL = Path(__file__).parent / "vision_piece_dataset" / "models" / "piece_random_forest.joblib"


class LearnedPieceClassifier:
    def __init__(
        self,
        model_path=None,
        min_confidence=0.35,
        debug=False,
    ):
        self.model_path = Path(model_path) if model_path is not None else _DEFAULT_MODEL
        self.min_confidence = min_confidence
        self.debug = debug

        payload = joblib.load(self.model_path)
        self.model = payload["model"]
        self.labels = payload["labels"]

    def classify(self, circle_bgr):
        if circle_bgr is None or circle_bgr.size == 0:
            return Cell.EMPTY

        x = extract_features(circle_bgr).reshape(1, -1)

        pred_id = int(self.model.predict(x)[0])
        probs = self.model.predict_proba(x)[0]
        confidence = float(np.max(probs))

        label = self.labels[pred_id]

        if self.debug:
            prob_str = {
                self.labels[int(cls_id)]: float(prob)
                for cls_id, prob in zip(self.model.classes_, probs)
            }
            print(f"[LearnedClassifier] label={label}, conf={confidence:.2f}, probs={prob_str}")

        if confidence < self.min_confidence:
            return Cell.EMPTY

        if label == "red":
            return Cell.ROBOT

        if label == "yellow":
            return Cell.HUMAN

        return Cell.EMPTY