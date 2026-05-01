from pathlib import Path
from PIL import Image

src = Path(".")
dst = src / "resized_tags_15mm"
dst.mkdir(exist_ok=True)

files = sorted(src.glob("tag36_11_*.png"))

for i, f in enumerate(files, start=1):
    img = Image.open(f)

    # 15 mm @ 600 DPI = 354 px
    img = img.resize((354, 354), Image.NEAREST)

    out_name = f"tag15mm_{i:02d}.png"
    img.save(dst / out_name)

    print("saved:", out_name)

print("\nDone.")
print("Output folder:", dst)