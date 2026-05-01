from PIL import Image

img = Image.open("tag36_11_00000.png")
img = img.resize((354,354), Image.NEAREST)   # for 15 mm @ 600 dpi
img.save("tag15mm.png")