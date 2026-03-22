import numpy as np
import cv2
from functions import render_img

# Load input data
data = np.load("hw1.npy", allow_pickle=True).item()
vertices = data["v_pos2d"]
vcolors = data["v_clr"]
faces = data["t_pos_idx"]
depth = data["depth"]

# Render with flat shading
img = render_img(faces, vertices, vcolors, uvs=None, depth=depth, shading="f", texImg=None)

# Save result
cv2.imwrite("flat_shading.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print("Saved flat_shading.png")
