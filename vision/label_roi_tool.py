from pathlib import Path
import shutil
import cv2


DATASET_DIR = Path("vision_piece_dataset")
UNLABELED_DIR = DATASET_DIR / "unlabeled"

LABEL_DIRS = {
    "r": DATASET_DIR / "red",
    "y": DATASET_DIR / "yellow",
    "e": DATASET_DIR / "empty",
}


def main():
    for d in LABEL_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    images = sorted(
        list(UNLABELED_DIR.glob("*.png"))
        + list(UNLABELED_DIR.glob("*.jpg"))
        + list(UNLABELED_DIR.glob("*.jpeg"))
    )

    if not images:
        print("No unlabeled images found.")
        return

    print("Controls:")
    print("  r = red / robot")
    print("  y = yellow / human")
    print("  e = empty")
    print("  s = skip")
    print("  q = quit")

    idx = 0

    while idx < len(images):
        path = images[idx]
        img = cv2.imread(str(path))

        if img is None:
            idx += 1
            continue

        display = cv2.resize(img, (220, 220), interpolation=cv2.INTER_NEAREST)

        cv2.putText(
            display,
            f"{idx + 1}/{len(images)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            display,
            path.name[:28],
            (10, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        cv2.imshow("Label ROI: r=red, y=yellow, e=empty, s=skip, q=quit", display)
        key = cv2.waitKey(0) & 0xFF

        char = chr(key).lower() if key < 128 else ""

        if char == "q":
            break

        if char == "s":
            idx += 1
            continue

        if char in LABEL_DIRS:
            dst = LABEL_DIRS[char] / path.name
            shutil.move(str(path), str(dst))
            print(f"{path.name} -> {dst.parent.name}")
            idx += 1
            continue

        print("Unknown key. Use r, y, e, s, q.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()