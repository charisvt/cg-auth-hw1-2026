import numpy as np
import cv2
from functions import render_img

# Load input data
data = np.load("hw1.npy", allow_pickle=True).item()
vertices = data["v_pos2d"]
uvs = data["v_uvs"]
vcolors = data["v_clr"]
faces = data["t_pos_idx"]
depth = data["depth"]

# Load texture image and normalize to [0, 1]
texture = cv2.imread("loony-repeat.png")
texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
texture = texture.astype(np.float32) / 255.0

# Render with texture shading
img = render_img(faces, vertices, vcolors, uvs=uvs, depth=depth, shading="t", texImg=texture)

# Save result
cv2.imwrite("texture_shading.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print("Saved texture_shading.png")
